"""This defines a reader for Arepo Friends-of-Friends (FoF) and subhalo catalog files"""
import numpy as np
import h5py
from .paicos_readers import PaicosReader
from ..writers.paicos_writer import PaicosWriter
from .. import settings
import numbers
import warnings


class PaicosDict(dict):
    def __init__(self, cat, subfind_catalog=True):

        self.cat = cat
        group_keys, sub_keys = self.cat.info(verbose=False)
        if subfind_catalog:
            self.loader = self.cat.load_sub_data
            self._auto_list = sub_keys
        else:
            self.loader = self.cat.load_group_data
            self._auto_list = group_keys

    def __getitem__(self, ikey):
        self.loader(ikey)
        return super().__getitem__(ikey)

    def _ipython_key_completions_(self):
        """
        Auto-completion of dictionary.
        """

        return self._auto_list


class Catalog(PaicosReader):
    """
    This is a Python class for reading Arepo group and subhalo catalogs.
    The class is based on a script originally written by Ewald Puchwein,
    which has since then been modified and included in Paicos.

    The class takes in the path of the directory containing the catalog, the
    catalog number, and an optional basename parameter, and uses this
    information to locate and open the catalog files. The class also loads
    the catalog's header, parameters, and configuration.
    The class also includes methods to extract the redshift, scale
    factor, and other properties of the catalog.

    Important methods and attributes:
    ---------------------------------

        cat = Catalog()

        cat.Group : dict
            Contains a dictionary of the FoF-catalog.

        cat.Sub : dict
            Contains a dictionary of the Subfind-catalog.

        cat.Parameters : dict
            Contains information from the parameter file used in the simulation (e.g. param.txt).

        cat.Config : dict
            Contains information from the Config file used in the simulation (e.g. Config.txt).

        cat.Header : dict
            Contains information about this particular catalog such as its time (e.g scale factor).

        cat.z : float
            The redshift.

        cat.h : float
            Reduced Hubble param (e.g. 0.67).

        cat.age : float
            The age of the Universe (only for cosmological runs).

        cat.lookback_time : float
            The age of the Universe (only for cosmological runs).

    """
    def __init__(self, basedir='.', snapnum=None, load_all=False,
                 to_physical=False, subfind_catalog=True, readonly_first_file=False,
                 verbose=False):
        """
        Initializes the Catalog class.

        Parameters
        ----------

            basedir : str
                The path of the directory containing the catalogs
                (e.g. the 'output' folder).

            snapnum : int
                The snapshot number.

            load_all : bool
                Whether to immediately load all fields or not.

            to_physical : bool
                whether to convert to physical unit upon loading the data.
                Default is False.

            subfind_catalog : bool
                whether the simulation has subfind catalogs,
                when False the code will look for FoF-catalogs only.

            readonly_first_file : bool
                when set to True the code will only read one catalog file (the first).
                This can significantly speed things up when one is only interested in the
                properties of the most massive groups in a simulation.

            verbose : bool
                whether to print information, default is False.

        """

        if subfind_catalog:
            basename = 'fof_subhalo_tab'
        else:
            basename = 'fof_tab'

        super().__init__(basedir=basedir, snapnum=snapnum, basename=basename,
                         load_all=False, to_physical=to_physical,
                         basesubdir='groups', verbose=verbose)

        # give names to specific header fields
        self.nfiles = self.Header["NumFiles"]
        self.ngroups = self.Header["Ngroups_Total"]

        if "Nsubgroups_Total" in self.Header.keys():
            self.nsubs = self.Header["Nsubgroups_Total"]
        else:
            self.nsubs = self.Header["Nsubhalos_Total"]

        # Initialize dictionaries
        self.Group = {}
        self.Sub = {}

        self.Group = PaicosDict(self, subfind_catalog=False)
        self.Sub = PaicosDict(self, subfind_catalog=True)

        self.to_physical = to_physical

        self.readonly_first_file = readonly_first_file

        if to_physical and not settings.use_units:
            err_msg = "to_physical=True requires that units are enabled"
            raise RuntimeError(err_msg)

        # Load all data
        if load_all:
            self.load_all_data()

    def info(self, verbose=True):
        """
        Return the available keys for the FoF catalog and/or
        the subhalo catalo.

        :meta private:
        """
        ifile = 0
        if self.multi_file is False:
            cur_filename = self.filename
        else:
            if self.no_subdir:
                cur_filename = self.multi_wo_dir.format(ifile)
            else:
                cur_filename = self.multi_filename.format(ifile)

        if self.verbose:
            print("reading file", cur_filename)

        file = h5py.File(cur_filename, "r")

        if "Group" in file:
            group_keys = list(file["Group"].keys())
        else:
            group_keys = []

        if "Subhalo" in file:
            sub_keys = list(file["Subhalo"].keys())
        else:
            sub_keys = []

        file.close()

        if not settings.use_units:
            if verbose:
                if len(group_keys) > 0:
                    print('Available group_keys are:', group_keys)

                if len(sub_keys) > 0:
                    print('\nAvailable sub_keys are:', sub_keys)
            return group_keys, sub_keys
        else:
            from .. import unit_specifications

        not_implemented_group_keys = []
        implemented_group_keys = []
        for key in list(group_keys):
            if key in unit_specifications.unit_dict['groups']:
                implemented_group_keys.append(key)
            else:
                not_implemented_group_keys.append(key)

        not_implemented_sub_keys = []
        implemented_sub_keys = []
        for key in list(sub_keys):
            if key in unit_specifications.unit_dict['subhalos']:
                implemented_sub_keys.append(key)
            else:
                not_implemented_sub_keys.append(key)

        if verbose:
            if len(group_keys) > 0:
                print('Available group_keys are:', implemented_group_keys)
                if len(not_implemented_group_keys) > 0:
                    print('\nNot implemented group_keys are:', not_implemented_group_keys)

            if len(sub_keys) > 0:
                print('\nAvailable sub_keys are:', implemented_sub_keys)
                if len(not_implemented_sub_keys) > 0:
                    print('\nNot implemented sub_keys are:', not_implemented_sub_keys)

        return implemented_group_keys, implemented_sub_keys

    def load_data(self):
        """
        Overwrite of the base class method.

        :meta private:
        """
        raise RuntimeError('should not be called')

    def load_group_data(self, ikey):
        """
        Load subhalo dato. See self.info for valid ikeys

        :meta private:
        """

        if ikey in self.Group:
            return

        data = None  # to get rid of linting error

        if not self.readonly_first_file:
            skip_gr = 0

            for ifile in range(self.nfiles):
                if self.multi_file is False:
                    cur_filename = self.filename
                else:
                    if self.no_subdir:
                        cur_filename = self.multi_wo_dir.format(ifile)
                    else:
                        cur_filename = self.multi_filename.format(ifile)

                if self.verbose:
                    print("reading file", cur_filename)

                file = h5py.File(cur_filename, "r")

                ng = int(file["Header"].attrs["Ngroups_ThisFile"])

                # initialize arrays
                if ifile == 0:
                    if "Group" in file:
                        if len(file["Group/" + ikey].shape) == 1:
                            data = np.empty(
                                self.ngroups, dtype=file["Group/" + ikey].dtype)
                        elif len(file["Group/" + ikey].shape) == 2:
                            data = np.empty(
                                (self.ngroups, file["Group/" + ikey].shape[1]),
                                dtype=file["Group/" + ikey].dtype)
                        else:
                            assert False

                # read group data
                if ng > 0:
                    data[skip_gr:skip_gr + ng] = file["Group/" + ikey]

                skip_gr += ng

                file.close()
        else:
            cur_filename = self.filename
            file = h5py.File(cur_filename, "r")
            ng = int(file["Header"].attrs["Ngroups_ThisFile"])
            if "Group" in file:
                if len(file["Group/" + ikey].shape) == 1:
                    data = np.empty(
                        ng, dtype=file["Group/" + ikey].dtype)
                elif len(file["Group/" + ikey].shape) == 2:
                    data = np.empty(
                        (ng, file["Group/" + ikey].shape[1]),
                        dtype=file["Group/" + ikey].dtype)
                else:
                    assert False

            # read group data
            if ng > 0:
                data[:ng] = file["Group/" + ikey][...]

            file.close()

        # Load all variables with double precision
        if settings.double_precision:
            if not issubclass(data.dtype.type, numbers.Integral):
                data = data.astype(np.float64)

        else:
            warnings.warn('\n\nThe cython routines expect double precision '
                          + 'and will fail unless settings.double_precision '
                          + 'is True.\n\n')

        if settings.use_units:
            self.Group[ikey] = self.get_paicos_quantity(
                data, ikey,
                field='groups')
            if not hasattr(self.Group[ikey], 'unit'):
                del self.Group[ikey]
                raise RuntimeError(f"{ikey} does not have units implemented!")
        else:
            self.Group[ikey] = data

    def load_sub_data(self, ikey):
        """
        Load subhalo dato. See self.info for valid ikeys

        TODO: Merge with load_group_data, lot's of duplicate code here...

        :meta private:
        """

        if ikey in self.Sub:
            return

        data = None  # to get rid of linting error

        skip_sub = 0
        for ifile in range(self.nfiles):
            if self.multi_file is False:
                cur_filename = self.filename
            else:
                if self.no_subdir:
                    cur_filename = self.multi_wo_dir.format(ifile)
                else:
                    cur_filename = self.multi_filename.format(ifile)

            if self.verbose:
                print("reading file", cur_filename)

            file = h5py.File(cur_filename, "r")

            if "Nsubgroups_ThisFile" in file["Header"].attrs.keys():
                ns = int(file["Header"].attrs["Nsubgroups_ThisFile"])
            else:
                ns = int(file["Header"].attrs["Nsubhalos_ThisFile"])

            # initialize arrays
            if ifile == 0:
                if "Subhalo" in file:
                    if len(file["Subhalo/" + ikey].shape) == 1:
                        data = np.empty(
                            self.nsubs, dtype=file["Subhalo/" + ikey].dtype)
                    elif len(file["Subhalo/" + ikey].shape) == 2:
                        data = np.empty(
                            (self.nsubs, file["Subhalo/" + ikey].shape[1]),
                            dtype=file["Subhalo/" + ikey].dtype)
                    else:
                        assert False

            # read subhalo data
            if ns > 0:
                data[skip_sub:skip_sub + ns] = file["Subhalo/" + ikey]

            skip_sub += ns

            file.close()

        # Load all variables with double precision
        if settings.double_precision:
            if not issubclass(data.dtype.type, numbers.Integral):
                data = data.astype(np.float64)
        else:
            warnings.warn('\n\nThe cython routines expect double precision '
                          + 'and will fail unless settings.double_precision '
                          + 'is True.\n\n')

        if settings.use_units:
            self.Sub[ikey] = self.get_paicos_quantity(
                data, ikey,
                field='subhalos')
            if not hasattr(self.Sub[ikey], 'unit'):
                del self.Sub[ikey]
                raise RuntimeError(f"{ikey} does not have units implemented!")
        else:
            self.Sub[ikey] = data

    def load_all_data(self):
        """
        Calling this method simply loads all the data in the catalog.
        """

        for ikey in self.Group._auto_list:
            self.load_group_data(ikey)

        for ikey in self.Sub._auto_list:
            self.load_sub_data(ikey)

    def save_new_catalog(self, basename, basedir=None, single_precision=False):
        """
        Save a new catalog containing only the currently loaded
        variables. Useful for reducing datasets to smaller sizes.
        """
        if basedir is None:
            writer = PaicosWriter(self, self.basedir, basename, 'w')
        else:
            writer = PaicosWriter(self, basedir, basename, 'w')

        for key in self.Group:
            Ngroups_Total = self.Group[key].shape[0]

            if single_precision:
                data = self.Group[key].astype(np.float32)
            else:
                data = self.Group[key]
            writer.write_data(key, data, group='Group')

        for key in self.Sub:
            Nsubgroups_Total = self.Sub[key].shape[0]

            if single_precision:
                data = self.Sub[key].astype(np.float32)
            else:
                data = self.Sub[key]
            writer.write_data(key, data, group='Subhalo')

        with h5py.File(writer.tmp_filename, 'r+') as f:
            f['Header'].attrs["Ngroups_ThisFile"] = Ngroups_Total
            f['Header'].attrs["Ngroups_Total"] = Ngroups_Total
            f['Header'].attrs["Nsubgroups_ThisFile"] = Nsubgroups_Total
            f['Header'].attrs["Nsubgroups_Total"] = Nsubgroups_Total
            f['Header'].attrs["NumFiles"] = 1

        writer.finalize()

        return writer
