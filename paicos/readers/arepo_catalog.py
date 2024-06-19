"""This defines a reader for Arepo Friends-of-Friends (FoF) and subhalo catalog files"""
import numpy as np
import h5py
from .paicos_readers import PaicosReader
from ..writers.paicos_writer import PaicosWriter
from .. import settings
import numbers
import warnings


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
    def __init__(self, basedir='.', snapnum=None, load_all=True,
                 to_physical=False, subfind_catalog=True, verbose=False):
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

        # Load all data
        if load_all:
            self.load_all_data()

    def load_data(self):
        """
        Overwrite of the base class method.

        :meta private:
        """
        pass

    def load_all_data(self):
        """
        Calling this method simply loads all the data in the catalog.

        TODO: For large catalogs it might be useful to implement on-demand
        access in a similar way to what we have for snapshots.
        """

        skip_gr = 0
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

            ng = int(file["Header"].attrs["Ngroups_ThisFile"])

            if "Nsubgroups_ThisFile" in file["Header"].attrs.keys():
                ns = int(file["Header"].attrs["Nsubgroups_ThisFile"])
            else:
                ns = int(file["Header"].attrs["Nsubhalos_ThisFile"])

            # initialize arrays
            if ifile == 0:
                if "Group" in file:
                    for ikey in file["Group"].keys():
                        if len(file["Group/" + ikey].shape) == 1:
                            self.Group[ikey] = np.empty(
                                self.ngroups, dtype=file["Group/" + ikey].dtype)
                        elif len(file["Group/" + ikey].shape) == 2:
                            self.Group[ikey] = np.empty(
                                (self.ngroups, file["Group/" + ikey].shape[1]),
                                dtype=file["Group/" + ikey].dtype)
                        else:
                            assert False
                if "Subhalo" in file:
                    for ikey in file["Subhalo"].keys():
                        if len(file["Subhalo/" + ikey].shape) == 1:
                            self.Sub[ikey] = np.empty(
                                self.nsubs, dtype=file["Subhalo/" + ikey].dtype)
                        elif len(file["Subhalo/" + ikey].shape) == 2:
                            self.Sub[ikey] = np.empty(
                                (self.nsubs, file["Subhalo/" + ikey].shape[1]),
                                dtype=file["Subhalo/" + ikey].dtype)
                        else:
                            assert False

            # read group data
            if ng > 0:
                for ikey in file["Group"].keys():
                    self.Group[ikey][skip_gr:skip_gr + ng] = file["Group/" + ikey]

            # read subhalo data
            if ns > 0:
                for ikey in file["Subhalo"].keys():
                    self.Sub[ikey][skip_sub:skip_sub + ns] = file["Subhalo/" + ikey]

            skip_gr += ng
            skip_sub += ns

            file.close()

        # Load all variables with double precision
        if settings.double_precision:

            for ikey in self.Group:
                if not issubclass(self.Group[ikey].dtype.type, numbers.Integral):
                    self.Group[ikey] = self.Group[ikey].astype(np.float64)

            for ikey in self.Sub:
                if not issubclass(self.Sub[ikey].dtype.type, numbers.Integral):
                    self.Sub[ikey] = self.Sub[ikey].astype(np.float64)
        else:
            warnings.warn('\n\nThe cython routines expect double precision '
                          + 'and will fail unless settings.double_precision '
                          + 'is True.\n\n')

        if settings.use_units:
            for key in list(self.Group.keys()):
                self.Group[key] = self.get_paicos_quantity(
                    self.Group[key], key,
                    field='groups')
                if not hasattr(self.Group[key], 'unit'):
                    del self.Group[key]

            for key in list(self.Sub.keys()):
                self.Sub[key] = self.get_paicos_quantity(
                    self.Sub[key], key,
                    field='subhalos')
                if not hasattr(self.Sub[key], 'unit'):
                    del self.Sub[key]

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
