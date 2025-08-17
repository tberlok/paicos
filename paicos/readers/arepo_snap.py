"""This defines a reader for Arepo snapshot files"""
import time
from inspect import signature
import numbers
import warnings
import numpy as np
import h5py
from .arepo_catalog import Catalog
from .paicos_readers import PaicosReader
from ..writers.paicos_writer import PaicosWriter
from .. import settings
from ..derived_variables import derived_variables


class Snapshot(PaicosReader):
    """
    This is a Python class for reading Arepo snapshots, which are simulations
    of the evolution of the universe using a code called Arepo. The class is
    based on a script originally written by Ewald Puchwein, which has since
    then been modified and included in Paicos.

    The class takes in the path of the directory containing the snapshot, the
    snapshot number, and an optional basename parameter, and uses this
    information to locate and open the snapshot files. The class also loads
    the snapshot's header, parameters, and configuration, and uses them to
    create a converter object that can be used to convert units in the
    snapshot. The class also includes methods to extract the redshift, scale
    factor, and other properties of the snapshot, as well as the subfind
    catalog if present.

    Important methods and attributes:
    ---------------------------------

        snap = Snapshot()

        snap.Group : dict
            Contains a dictionary of the FoF-catalog.

        snap.Sub : dict
            Contains a dictionary of the Subfind-catalog.

        snap.Parameters : dict
            Contains information from the parameter file used in the simulation (e.g. param.txt).

        snap.Config : dict
            Contains information from the Config file used in the simulation (e.g. Config.txt).

        snap.Header : dict
            Contains information about this particular catalog such as its time (e.g scale factor).

        snap.z : float
            The redshift.

        snap.h : float
            Reduced Hubble param (e.g. 0.67).

        snap.age : float
            The age of the Universe (only for cosmological runs).

        snap.lookback_time : float
            The age of the Universe (only for cosmological runs).

        snap.time : float
            The time stamp of the snapshot (only for non-cosmological runs).

        snap.box_size : array with length 3
            The dimensions of the simulation domain.

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, basedir, snapnum=None, basename="snap", load_all=False,
                 to_physical=False, load_catalog=None, subhalonum=None, fofnum=None,
                 verbose=False):
        """
        Initialize the Snapshot class.

        Parameters:

            basedir (str): path of the directory containing the snapshot
                           (e.g. the 'output' folder).

            snapnum (int): snapshot number.

            basename (str): name of the snapshot file, default is "snap".

            verbose (bool): whether to print information about the snapshot,
                            default is False.

            no_snapdir (bool): whether there is no snap directory, i.e.,
                               default is False.

            load_catalog (bool): whether to load the subfind catalog.
                                 The default None is internally changed to
                                 True for comoving simulations and to
                                 False for non-comoving simulations.

            subhalonum (int): optional parameter to only load a single subhalo.

            fofnum (int): optional parameter to only load a single FoF group.
        """

        super().__init__(basedir=basedir, snapnum=snapnum, basename=basename,
                         load_all=load_all, to_physical=to_physical,
                         basesubdir='snapdir', verbose=verbose)

        assert self.snapnum is not None

        self.load_catalog = load_catalog

        if not hasattr(self, "dic_selection_index"):
            self.dic_selection_index = {}

        self.nfiles = self.Header["NumFilesPerSnapshot"]
        self.npart = self.Header["NumPart_Total"]
        self.nspecies = self.npart.size

        if self.verbose:
            print("has", self.nspecies, "particle types")
            print("with npart =", self.npart)

        self.box = self.Header["BoxSize"]
        box_size = [self.box, self.box, self.box]

        for ii, dim in enumerate(['X', 'Y', 'Z']):
            if 'LONG_' + dim in self.Config:
                box_size[ii] *= self.Config['LONG_' + dim]

        if settings.use_units:
            get_paicos_quantity = self.get_paicos_quantity
            self.box_size = get_paicos_quantity(box_size, 'Coordinates')
            self.masstable = get_paicos_quantity(self.Header["MassTable"],
                                                 'Masses')
        else:
            self.box_size = np.array(box_size)
            self.masstable = self.Header["MassTable"]

        # get subfind catalog?
        if load_catalog is None:
            load_catalog = bool(self.ComovingIntegrationOn)

        if load_catalog:
            try:
                self.Cat = Catalog(
                    self.basedir, self.snapnum, verbose=self.verbose,
                    subfind_catalog=True)
            except FileNotFoundError:
                self.Cat = None

                if subhalonum is not None:
                    err_msg = f'No subhalo catalog found, cannot load subhalo={subhalonum}'
                    raise RuntimeError(err_msg)

            # If no subfind catalog found, then try for a fof catalog
            if self.Cat is None:
                try:
                    self.Cat = Catalog(
                        self.basedir, self.snapnum, verbose=self.verbose,
                        subfind_catalog=False)
                except FileNotFoundError:
                    warnings.warn('no catalog found')

                    if subhalonum is not None:
                        err_msg = f'No FoF catalog found, cannot load fof={fofnum}'
                        raise RuntimeError(err_msg)

        if fofnum is not None and subhalonum is not None:
            err_msg = f'Cannot select both fofnum={fofnum} and subhalonum={subhalonum}'
            raise RuntimeError(err_msg)
        elif fofnum is None and subhalonum is None:
            self.subselection = False
        else:
            self.subselection = True

        if self.subselection and not load_catalog:
            raise RuntimeError('Catalog needed!')

        self.subhalonum = subhalonum
        self.fofnum = fofnum
        if self.subselection:
            if self.subhalonum is not None:
                self._len_subfofs = self.Cat.Sub['SubhaloLenType'][subhalonum]
                if 'SubhaloOffsetType' in self.Cat.Sub:
                    self._first_indices = self.Cat.Sub['SubhaloOffsetType'][subhalonum]
                    self._last_indices = self.Cat.Sub['SubhaloOffsetType'][subhalonum + 1]
                else:
                    self._first_indices = np.sum(self.Cat.Sub['SubhaloLenType'][0:subhalonum], axis=0)
                    self._last_indices = np.sum(self.Cat.Sub['SubhaloLenType'][0:subhalonum + 1], axis=0)
            else:
                self._len_subfofs = self.Cat.Group['GroupLenType'][fofnum]
                if 'GroupOffsetType' in self.Cat.Group:
                    self._first_indices = self.Cat.Group['GroupOffsetType'][fofnum]
                    self._last_indices = self.Cat.Group['GroupOffsetType'][fofnum + 1]
                else:
                    self._first_indices = np.sum(self.Cat.Group['GroupLenType'][0:fofnum], axis=0)
                    self._last_indices = np.sum(self.Cat.Group['GroupLenType'][0:fofnum + 1], axis=0)

        self.P_attrs = {}  # attributes

        self.derived_data_counter = 0

        self._add_mass_to_user_funcs()

        self._find_available_for_loading()
        self._find_available_functions()

        self._identify_parttypes()

        self.__get_auto_comple_list()

    def _add_mass_to_user_funcs(self):
        """
        This functions adds functionality for obtaining the masses
        of particle types which do not have the Masses blockname stored.
        These normally instead have their mass in masstable.
        """

        self._this_snap_funcs = {}

        class Mass:
            """
            This class allows us to get a function which is just a function of
            one parameter.
            """
            def __init__(self, parttype):
                """
                The parttype, e.g. 1 for DM particles
                """
                self.parttype = parttype

            def get_masses_from_header(self, snap):
                """
                Get mass of particle type from the mass table
                """
                parttype = self.parttype
                if parttype in snap.dic_selection_index:
                    npart = snap.dic_selection_index[parttype].shape[0]
                else:
                    if not snap.subselection:
                        npart = snap.npart[parttype]
                    else:
                        npart = snap._len_subfofs[parttype]
                return np.ones(npart) * snap.masstable[parttype]

        # Add function to the ones available
        for parttype in range(self.nspecies):
            if self.masstable[parttype] != 0:
                p_key = str(parttype) + '_Masses'
                obj = Mass(parttype)
                self._this_snap_funcs[p_key] = obj.get_masses_from_header

    def _find_available_for_loading(self):
        """
        Read the hdf5 file info and find all the blocknames
        that are available for each particle type.
        """
        self._all_avail_load = []
        self._part_avail_load = {i: [] for i in range(self.nspecies)}
        self._part_specs = {i: {} for i in range(self.nspecies)}
        for parttype in range(self.nspecies):
            parttype_str = f'PartType{parttype}'

            if not self.multi_file:
                with h5py.File(self.filename, 'r') as file:
                    if parttype_str in file:
                        for key in file[parttype_str]:
                            p_key = f'{parttype}_{key}'
                            self._part_avail_load[parttype].append(p_key)
                            shape = file[parttype_str][key].shape
                            dtype = file[parttype_str][key].dtype
                            self._part_specs[parttype][key] = {'shape': shape,
                                                               'dtype': dtype}
            else:
                # For multiple files we sometimes need to open many of them
                # to find *all* the available parttypes.
                # This can occur when the parttypes are not evenly distributed
                # in space (e.g. the high resolution region in a zoom
                # will have no low res DM particles).
                for ii in range(self.nfiles):
                    with h5py.File(self.multi_filename.format(ii), 'r') as file:
                        if parttype_str in file:
                            for key in file[parttype_str]:
                                p_key = f'{parttype}_{key}'
                                self._part_avail_load[parttype].append(p_key)
                                shape = list(file[parttype_str][key].shape)
                                if len(shape) > 1:
                                    shape[0] = self.npart[parttype]
                                dtype = file[parttype_str][key].dtype
                                self._part_specs[parttype][key] = {'shape': tuple(shape),
                                                                   'dtype': dtype}
                            # print(f'found parttype in partfile {ii}, breaking out')
                            break

            self._all_avail_load += self._part_avail_load[parttype]

    def _identify_parttypes(self):
        """
        Try to figure out which physical variable is stored in each
        particle type.
        """
        self._type_info = {0: 'voronoi_cells'}
        for p in range(1, self.nspecies):
            bh = any('BH_' in key for key in self._part_avail_load[p])
            star = any('GFM_' in key for key in self._part_avail_load[p])
            if bh:
                self._type_info[p] = 'black_holes'
            if star:
                self._type_info[p] = 'stars'

    def _find_available_functions(self):
        """
        This function goes through all the implemented functions
        for getting derived variables. Checking their dependencies,
        it then figures out which derived variables are actually possible
        for this particular snapshot. For instance, you cannot calculate the
        magnetic field strength if the magnetic field is not stored in
        the hdf5 file.
        """

        user_functs = derived_variables.user_functions

        for func_name, func in user_functs.items():
            self._this_snap_funcs[func_name] = func

        # Add all implemented functions
        if not settings.use_only_user_functions:
            def_functs = derived_variables.default_functions
            for func_name, func in def_functs.items():
                if func_name not in self._this_snap_funcs:
                    self._this_snap_funcs[func_name] = func

        # Build a dependency dictionary by asking each function for its
        # dependencies
        self._dependency_dic = {}
        for func_name, func in self._this_snap_funcs.items():
            sig = signature(func)
            # If the function has two input arguments, then we assume
            # that passing True to the second argument will return its
            # dependencies
            if len(sig.parameters) == 2:
                self._dependency_dic[func_name] = func(self, True)
            else:
                self._dependency_dic[func_name] = []

        dependency_dic = dict(self._dependency_dic)

        # We then trim the dependency dictionary

        # This will fail for very nested dependencies.
        for _ in range(4):
            # First substitute all dependencies that are
            # at the top level of the dictionary
            for func_name, deps in dependency_dic.items():
                for dep in list(deps):
                    if dep in dependency_dic:
                        deps.remove(dep)
                        for subdep in dependency_dic[dep]:
                            deps.append(subdep)

            # Then remove all the dependencies that can be loaded
            for func_name, deps in dependency_dic.items():
                for dep in list(deps):
                    if dep in self._all_avail_load:
                        deps.remove(dep)

        # Delete the entries where we do not have the requirements
        for func_name, deps in dependency_dic.items():
            if len(deps) > 0:
                if func_name in user_functs:
                    msg = (f'Deleting the user function: {user_functs[func_name]} '
                           + f'because its dependency: {deps} '
                           + 'is missing')
                    warnings.warn(msg)
                del self._this_snap_funcs[func_name]

    def get_variable_function(self, p_key, info=False):
        """
        This is a helper function for 'get_derived_data'. It returns a
        function if info is False, and a list of all available
        functions for parttype = p_key[0] if info is True.

        :meta private:
        """

        assert isinstance(p_key, str)

        if not p_key[0].isnumeric() or p_key[1] != '_':
            msg = (f"The key '{p_key}' is not valid."
                   + "\n\nKeys are expected to consist of an integer "
                   + "(the particle type) and a blockname, separated by a "
                   + " _. For instance 0_Density. You can get the "
                   + "available fields like so: snap.info(0)")
            raise RuntimeError(msg)

        # Return a list with all available keys for this parttype
        if info:
            parttype = int(p_key[0])
            avail_list = []
            for key in self._this_snap_funcs:
                if int(key[0]) == parttype:
                    avail_list.append(key)

            return avail_list

        # Return a function
        if p_key in self._this_snap_funcs:
            return self._this_snap_funcs[p_key]
        msg = '\n\n{} not found in the functions: {}'
        msg = msg.format(p_key, self._this_snap_funcs)
        raise RuntimeError(msg)

    def info(self, parttype, verbose=True):
        """
        This function provides information about the keys of a certain
        particle type in a snapshot.

        Parameters
        -----------

            partType (int):
                           An integer representing the particle type of
                           interest, e.g. 0 for gas, 1 for DM etc.

            verbose (bool):
                           A flag indicating whether or not
                           to print the keys to the console.
                           Defaults to True.

        Returns
        -------

            list or None :
                          If the requested PartType exists in the file, a list of
                          keys for that PartType is returned, otherwise None.
        """
        parttype_str = f'PartType{parttype}'
        if parttype >= self.nspecies:
            err_msg = (f"Parttype {self.nspecies-1} is the largest contained"
                       + f" in the snapshot which has nspecies={self.nspecies}.")
            raise RuntimeError(err_msg)

        load_keys = self._part_avail_load[parttype]
        if verbose:
            print('\nKeys for ' + parttype_str + ' in the hdf5 file:')
            for key in (sorted(load_keys)):
                print_key = key
                if settings.use_aliases:
                    if key in settings.aliases:
                        alias = settings.aliases[key]
                        msg = alias + '\t' * 5 + '(an alias of {})'
                        print_key = msg.format(key)

                print(print_key)

            print('\nPossible derived variables are:')
            dkeys = self.get_variable_function(f'{parttype}_', True)

            for key in (sorted(dkeys)):
                print_key = key
                if settings.use_aliases:
                    if key in settings.aliases:
                        alias = settings.aliases[key]
                        msg = alias + '\t' * 5 + '(an alias of {})'
                        print_key = msg.format(key)

                print(print_key)
        else:
            return load_keys

    def _load_data_experimental(self, parttype, blockname):
        """
        Load data from hdf5 file(s). Example usage::

            snap = Snapshot(...)
            snap.load_data(0, 'Density')

        Note that subsequent calls does not reload the data. Reloading
        the data can be done explicitly::

            snap.remove_data(0, 'Density')
            snap.load_data(0, 'Density')

        """

        assert parttype < self.nspecies

        p_key = f'{parttype}_{blockname}'
        alias_key = p_key

        if settings.use_aliases:
            if p_key in settings.aliases:
                alias_key = settings.aliases[p_key]

        if p_key not in self.info(parttype, False):
            msg = (f'Unable to load parttype {parttype}, blockname '
                   + f'{blockname} as this field is not in the hdf5 file')
            raise RuntimeError(msg)

        if alias_key in self:
            if self.verbose:
                print(blockname, "for species",
                      parttype, "already in memory")
            return

        if self.verbose:
            print("loading block", blockname,
                  "for species", parttype, "...")
            start_time = time.time()

        def read_hdf5_file(filename, parttype, blockname):
            """
            This helper function does the actual data loading.

            :meta private:
            """
            parttype_str = f'PartType{parttype}'
            with h5py.File(filename, 'r') as f:
                # Load a dataset
                dataset = f[parttype_str][blockname][()]

            if settings.double_precision:
                # Load all variables with double precision
                if not issubclass(dataset.dtype.type, numbers.Integral):
                    dataset = dataset.astype(np.float64)
            else:
                warnings.warn('\n\nThe cython routines expect double precision '
                              + 'and will fail unless settings.double_precision '
                              + 'is True.\n\n')

            if settings.use_units:
                if parttype in self._type_info:
                    ptype = self._type_info[parttype]  # e.g. 'voronoi_cells'
                    dataset = self.get_paicos_quantity(dataset, blockname,
                                                       field=ptype)
                else:
                    # Assume dark matter for the units
                    dataset = self.get_paicos_quantity(dataset, blockname,
                                                       field='dark_matter')
            return dataset

        if self.multi_file:
            filenames = [self.multi_filename.format(ii) for ii in range(self.nfiles)]
            shape = self._part_specs[parttype][blockname]['shape']
            if len(shape) == 1:
                self[alias_key] = np.hstack([read_hdf5_file(filename, parttype, blockname)
                                             for filename in filenames])
            elif len(shape) == 2:
                self[alias_key] = np.vstack([read_hdf5_file(filename, parttype, blockname)
                                             for filename in filenames])
            else:
                raise RuntimeError(f"Unexpected shape={shape} for {parttype}_{blockname}")
        else:
            self[alias_key] = read_hdf5_file(self.filename, parttype, blockname)

        # Only keep the cells with True in the selection index array
        if parttype in self.dic_selection_index:
            selection_index = self.dic_selection_index[parttype]
            shape = self[alias_key].shape
            if len(shape) == 1:
                self[alias_key] = self[alias_key][selection_index]
            elif len(shape) == 2:
                self[alias_key] = self[alias_key][selection_index, :]
            else:
                raise RuntimeError('Data has unexpected shape!')

        if self.verbose:
            print("... done! (took", time.time() - start_time, "s)")

    def load_data(self, parttype, blockname, debug=False):
        """
        Load data from hdf5 file(s). Example usage::

            snap = Snapshot(...)
            snap.load_data(0, 'Density')

        Note that subsequent calls does not reload the data. Reloading
        the data can be done explicitly::

            snap.remove_data(0, 'Density')
            snap.load_data(0, 'Density')
        """

        assert parttype < self.nspecies

        p_key = f'{parttype}_{blockname}'
        alias_key = p_key

        if settings.use_aliases:
            if p_key in settings.aliases:
                alias_key = settings.aliases[p_key]

        if p_key not in self.info(parttype, False):
            msg = (f'Unable to load parttype {parttype}, blockname '
                   + f'{blockname} as this field is not in the hdf5 file')
            raise RuntimeError(msg)

        datname = f'PartType{parttype}/{blockname}'
        if alias_key in self:
            if self.verbose:
                print(blockname, "for species",
                      parttype, "already in memory")
            return

        if self.verbose:
            print("loading block", blockname,
                  "for species", parttype, "...")
            start_time = time.time()

        skip_part = 0
        if self.subselection:
            subfof_skip_part = 0

        # Initialize array
        if not self.subselection:
            length = self.npart[parttype]
        else:
            length = len_subfof = self._len_subfofs[parttype]

        if length == 0:
            err_msg = f'{p_key} has zero length, not able to load anything'
            RuntimeError(err_msg)

        shape = self._part_specs[parttype][blockname]['shape']
        dtype = self._part_specs[parttype][blockname]['dtype']
        if len(shape) == 1:
            self[alias_key] = np.empty(length, dtype=dtype)
        else:
            self[alias_key] = np.empty((length, shape[1]), dtype=dtype)

        for ifile in range(self.nfiles):
            if self.multi_file is False:
                cur_filename = self.filename
            else:
                if self.no_subdir:
                    cur_filename = self.multi_wo_dir.format(ifile)
                else:
                    cur_filename = self.multi_filename.format(ifile)

            f = h5py.File(cur_filename, "r")

            if 'unit' in f[datname].attrs.keys():
                unit_str = f[datname].attrs['unit']
            else:
                unit_str = None

            np_file = int(f["Header"].attrs["NumPart_ThisFile"][parttype])  # pylint: disable=unsubscriptable-object

            if not self.subselection:
                if np_file > 0:
                    self[alias_key][skip_part:skip_part + np_file] = f[datname][...]
            else:
                # Global indices
                file_first = skip_part
                file_last = skip_part + np_file
                subfof_first = self._first_indices[parttype]
                subfof_last = self._last_indices[parttype]
                found_something = False
                if subfof_first >= file_first and subfof_first < file_last:
                    found_something = True
                    first = subfof_first - file_first
                    last = np.min([np_file, subfof_last - file_first])
                elif subfof_last > file_first and subfof_last < file_last:
                    found_something = True
                    first = 0
                    last = np.min([np_file, subfof_last - file_first])
                elif subfof_first <= file_first and subfof_last >= file_last:
                    found_something = True
                    first = 0
                    last = np_file
                else:
                    found_something = False
                    if debug:
                        pass
                if found_something:
                    np_file_subfof = last - first
                    if debug:
                        if self.subhalonum is not None:
                            print(f'ifile={ifile}, subhalo {self.subhalonum} in file')
                        else:
                            print(f'ifile={ifile}, fofnum {self.fofnum} in file')
                        print('file_first, file_last', file_first, file_last)
                        print('(subfof_first, subfof_last', subfof_first, subfof_last)
                        print('np_file, np_file_subfof', np_file, np_file_subfof)
                        print("len_subfof", len_subfof)

                    self[alias_key][subfof_skip_part:subfof_skip_part + np_file_subfof] = f[datname][first:last]
                    subfof_skip_part += np_file_subfof

            skip_part += np_file

            if debug:
                if self.subselection:
                    print(subfof_skip_part, length)
                else:
                    print(skip_part, length)

            f.close()

        if self.subselection:
            err_msg = f'missing particles!, {subfof_skip_part}/{length}'
            assert subfof_skip_part == length, err_msg
        else:
            err_msg = f'missing particles!, {skip_part}/{length}'
            assert skip_part == length, err_msg

        if settings.double_precision:
            # Load all variables with double precision
            if not issubclass(self[alias_key].dtype.type, numbers.Integral):
                self[alias_key] = self[alias_key].astype(np.float64)
        else:
            warnings.warn('\n\nThe cython routines expect double precision '
                          + 'and will fail unless settings.double_precision '
                          + 'is True.\n\n')

        # Only keep the cells with True in the selection index array
        if parttype in self.dic_selection_index:
            selection_index = self.dic_selection_index[parttype]
            shape = self[alias_key].shape
            if len(shape) == 1:
                self[alias_key] = self[alias_key][selection_index]
            elif len(shape) == 2:
                self[alias_key] = self[alias_key][selection_index, :]
            else:
                raise RuntimeError('Data has unexpected shape!')

        if settings.use_units:
            if unit_str is not None:
                self[alias_key] = self[alias_key] * self.unit_quantity(unit_str)
            else:
                if parttype in self._type_info:
                    ptype = self._type_info[parttype]  # e.g. 'voronoi_cells'
                    self[alias_key] = self.get_paicos_quantity(self[alias_key],
                                                               blockname,
                                                               field=ptype)
                else:
                    # Assume dark matter for the units
                    self[alias_key] = self.get_paicos_quantity(self[alias_key],
                                                               blockname,
                                                               field='dark_matter')
                if not hasattr(self[alias_key], 'unit'):
                    del self[alias_key]

        if self.verbose:
            print("... done! (took", time.time() - start_time, "s)")

    def get_derived_data(self, parttype, blockname, verbose=False):
        """
        Get derived quantities. Example usage:

        snap = Snapshot(...)

        snap.get_derived_data(0, 'Temperatures')

        :meta private:
        """
        # from .derived_variables import get_variable_function

        p_key = str(parttype) + "_" + blockname

        msg = (f'\n\n{blockname} is in the hdf5 file(s), please use load_data '
               + 'instead of get_derived_data')
        assert p_key not in self.info(parttype, False), msg

        if verbose:
            msg1 = f'Attempting to get derived variable: {p_key}...'
            msg2 = f'So we need the variable: {p_key}...'
            if self.derived_data_counter == 0:
                print(msg1, end='')
            else:
                print('\n\t' + msg2, end='')
            self.derived_data_counter += 1

        func = self.get_variable_function(p_key)

        if settings.use_aliases:
            if p_key in settings.aliases:
                p_key = settings.aliases[p_key]
        self[p_key] = func(self)

        if verbose:
            self.derived_data_counter -= 1
            if self.derived_data_counter == 0:
                print('\t[DONE]\n')

    def __getitem__(self, p_key):
        """
        This method is a special method in Python classes, known as a "magic
        method" that allows instances of the class to be accessed like a
        dictionary, using the bracket notation.

        This method is used to access the data stored in the class, it takes a
        single argument:

        p_key : a string that represents the data that is being accessed, it
        should be in the format of parttype_name, where parttype is an integer
        and name is the name of the data block. It first checks if the key is
        already in the class, if not it checks if the key is in the format of
        parttype_name and if the parttype exists in the file and the name is a
        valid key in the hdf5 file. If it is, the method loads the data from
        the hdf5 file, otherwise it calls the get_derived_data method to
        calculate the derived data. If the key does not meet the expected
        format or the parttype or data block does not exist it raises a
        RuntimeError with a message explaining the expected format.

        The method returns the value of the data block if it is found or
        loaded, otherwise it raises an error.

        This method allows for easy and convenient access to the data stored
        in the class, without the need for explicit method calls to load or
        calculate the data.

        That is, one can do snap['0_Density'] and the data will automatically
        be loaded.
        """

        if settings.use_aliases:

            if p_key in settings.inverse_aliases:
                p_key = settings.inverse_aliases[p_key]

        if p_key not in self:
            if not p_key[0].isnumeric() or p_key[1] != '_':
                msg = (f"The key '{p_key}' is not valid."
                       + "\n\nKeys are expected to consist of an integer "
                       + "(the particle type) and a blockname, separated by a "
                       + " _. For instance 0_Density. You can get the "
                       + "available fields like so: snap.info(0)")
                raise RuntimeError(msg)
            parttype = int(p_key[0])
            name = p_key[2:]

            if parttype >= self.nspecies:
                msg = f'Simulation only has {self.nspecies} species.'
                raise RuntimeError(msg)

            if p_key in self.info(parttype, False):
                self.load_data(parttype, name)
            else:
                verbose = settings.print_info_when_deriving_variables
                self.get_derived_data(parttype, name, verbose=verbose)

        if settings.use_aliases:
            if p_key in settings.aliases:
                p_key = settings.aliases[p_key]
        return super().__getitem__(p_key)

    def __get_auto_comple_list(self):
        """
        Pre-compute a list for auto-completion.
        """
        self._auto_list = list(self._all_avail_load)
        for key in self._this_snap_funcs:
            self._auto_list.append(key)

        if settings.use_aliases:
            for ii, p_key in enumerate(self._auto_list):
                if p_key in settings.aliases:
                    self._auto_list[ii] = settings.aliases[p_key]

    def _ipython_key_completions_(self):
        """
        Auto-completion of dictionary.
        """

        return self._auto_list

    def remove_data(self, parttype, blockname):
        """
        Remove data from object. Sometimes useful for for large datasets
        """
        p_key = str(parttype) + "_" + blockname
        if p_key in self:
            del self[p_key]
        if p_key in self.P_attrs:
            del self.P_attrs[p_key]

    def select(self, selection_index, parttype=None):
        """
        Create a new snapshot object which will only contain
        cells with a selection_index.

        Example use::

            index = snap['0_Density'] > snap['0_Density'].unit_quantity*1e-6
            selected_snap = snap.select(index, parttype=0)

        """
        if parttype is None:
            parttype = 0
            msg = ("You have not specified the parttype that the selection index"
                   + " corresponds to. The code will assume parttype=0. Future versions"
                   + " of this method will likely have parttype as a required input")
            warnings.warn(msg)

        s_index = selection_index

        if parttype in self.dic_selection_index:
            # This snap object is already a selection, combine the criteria!
            previous_selection = self.dic_selection_index[parttype]
            new_index = previous_selection[s_index]
            dic_selection_index = dict(self.dic_selection_index)
            dic_selection_index[parttype] = new_index
        else:
            # Convert to integer array
            if s_index.dtype == 'bool':
                s_index = np.arange(s_index.shape[0])[s_index]

            dic_selection_index = dict(self.dic_selection_index)
            dic_selection_index[parttype] = s_index

        select_snap = Snapshot(self.basedir, self.snapnum,
                               basename=self.basename,
                               verbose=self.verbose,
                               to_physical=self.to_physical,
                               load_catalog=self.load_catalog)

        select_snap.dic_selection_index = dic_selection_index

        for key in self:
            if key[0] == str(parttype):
                shape = self[key].shape
                if shape[0] == 1 and s_index.shape[0] != 1:
                    # Copy over single float numbers, e.g., constants,
                    # such as the mean molecular weight
                    select_snap[key] = self[key]
                else:
                    if len(shape) == 1:
                        select_snap[key] = self[key][s_index]
                    elif len(shape) == 2:
                        select_snap[key] = self[key][s_index, :]
                    else:
                        raise RuntimeError('Data has unexpected shape!')
            else:
                select_snap[key] = self[key]

        return select_snap

    def radial_select(self, center, r_max, r_min=0.0, parttype=None):
        """
        A convenience function for selecting in radius.

        Returns a new snapshot with the radial selection.

        :meta private:
        """
        from .. import util
        if parttype is None:

            index = util.get_index_of_radial_range(self['0_Coordinates'],
                                                   center, r_min, r_max)

            selected_snap = self.select(index, parttype=0)
            for p in range(1, self.nspecies):
                index = util.get_index_of_radial_range(selected_snap[f'{p}_Coordinates'],
                                                       center, r_min, r_max)
                selected_snap = selected_snap.select(index, parttype=p)

        else:
            p = parttype
            index = util.get_index_of_radial_range(self[f'{p}_Coordinates'],
                                                   center, r_min, r_max)
            selected_snap = self.select(index, parttype=p)

        return selected_snap

    def save_new_snapshot(self, basename, basedir=None, single_precision=False):
        """
        Save a new snapshot containing the currently loaded (derived)
        variables. Useful for reducing datasets to smaller sizes.
        """
        if basedir is None:
            writer = PaicosWriter(self, self.basedir, basename, 'w')
        else:
            writer = PaicosWriter(self, basedir, basename, 'w')

        new_npart = [0] * self.nspecies
        for key in self:
            for parttype in range(self.nspecies):
                if key[:2] == f'{parttype}_':
                    new_npart[parttype] = self[key].shape[0]
                    parttype_str = f'PartType{parttype}'

                    if single_precision:
                        data = self[key].astype(np.float32)
                    else:
                        data = self[key]
                    writer.write_data(key[2:], data, group=parttype_str)

        with h5py.File(writer.tmp_filename, 'r+') as f:
            f['Header'].attrs["NumFilesPerSnapshot"] = 1
            f['Header'].attrs["NumPart_Total"] = np.array(new_npart)
            f['Header'].attrs["NumPart_ThisFile"] = np.array(new_npart)

        writer.finalize()

        return writer

    def get_sum_of_array(self, variable):
        """
        Find the sum of a 1D array.

        Equivalent to np.sum(arr, axis=0) but with open_mp

        :meta private:
        """
        from .. import util

        if isinstance(variable, str):
            variable = self[variable]
        else:
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        if len(variable.shape) == 1:
            from ..cython.get_derived_variables import sum_1d_array_omp as sum_omp
        elif len(variable.shape) == 2:
            from ..cython.get_derived_variables import sum_2d_array_omp as sum_omp
        else:
            raise RuntimeError('unexpected')

        if settings.use_units:
            sum_omp = util.remove_astro_units(sum_omp)
            uq = variable.uq

            return sum_omp(variable, settings.numthreads_reduction) * uq
        return sum_omp(variable, settings.numthreads_reduction)

    def get_sum_of_array_times_vector(self, arr, vector):
        """
        Find the sum of a 1D array times a 2D array

        Equivalent to np.sum(arr[:, None] * vector, axis=0) but with open_mp

        :meta private:
        """
        from .. import util
        from ..cython.get_derived_variables import sum_arr_times_vector_omp
        numthreads = settings.numthreads_reduction

        if isinstance(arr, str):
            arr = self[arr]
        else:
            if not isinstance(arr, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        if isinstance(vector, str):
            vector = self[vector]
        else:
            if not isinstance(vector, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        assert vector.shape[1] == 3, 'only works on vectors, e.g. coordinates'

        if settings.use_units:
            sum_arr_times_vector_omp = util.remove_astro_units(sum_arr_times_vector_omp)
            uq = arr.uq * vector.uq

            return sum_arr_times_vector_omp(arr, vector, numthreads) * uq
        return sum_arr_times_vector_omp(arr, vector, numthreads)

    def get_sum_of_arr_times_vector_cross_product(self, mass, coord, velocity, center):
        """
        This code calculates sum_i (mass_i (coord_ij - center) x velocity_ij).

        That is, it returns the total angular momentum vector.

        :meta private:
        """
        from .. import util
        from ..cython.get_derived_variables import sum_arr_times_vector_cross_product
        numthreads = settings.numthreads_reduction

        if isinstance(mass, str):
            mass = self[mass]
        else:
            if not isinstance(mass, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        if isinstance(coord, str):
            coord = self[coord]
        else:
            if not isinstance(coord, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        if isinstance(velocity, str):
            velocity = self[velocity]
        else:
            if not isinstance(velocity, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        if settings.use_units:
            sum_func = util.remove_astro_units(sum_arr_times_vector_cross_product)
            uq = mass.uq * coord.uq * velocity.uq

            return sum_func(mass, coord, velocity, center, numthreads) * uq
        return sum_func(mass, coord, velocity, center, numthreads)

    def center_of_mass(self, parttype=None):
        """
        Finds the center of mass for the entire snapshot.

        parttype (default None):
                                 If None, then all parttypes are included in the
                                 center-of-mass calculation. If e.g. parttype=0,
                                 then the center of mass of the gas is returned.
                                 if parttype=None, then the total center and
                                 a list of of the parttype centers are returned

        This method can be used in combination with the select method to find the
        center of mass of a selection.
        """
        if parttype is None:
            mass = [self.get_sum_of_array(f'{p}_Masses') for p in range(self.nspecies)]
            mcord = [self.get_sum_of_array_times_vector(f'{p}_Masses', f'{p}_Coordinates')
                     for p in range(self.nspecies)]
            centers = [mcord[p] / mass[p] for p in range(self.nspecies)]

            tot_center = np.sum(np.vstack(mcord), axis=0) / np.sum(np.stack(mass))

            return tot_center, centers
        else:
            p = parttype
            mcord = self.get_sum_of_array_times_vector(f'{p}_Masses', f'{p}_Coordinates')
            mass = self.get_sum_of_array(f'{p}_Masses')
            return mcord / mass

    def total_angular_momentum(self, center, parttype=None):
        """
        Finds the total angular momentum for the entire snapshot.

        center: the center around which to calculate the angular momentum

        parttype (default None):
                                 If None, then all parttypes are included in the
                                 calculation. If e.g. parttype=0,
                                 then total angular momentum of the gas is returned.

        This method can be used in combination with the select method to find the
        total angular momentum of a selection.
        """
        func = self.get_sum_of_arr_times_vector_cross_product
        if parttype is None:
            if settings.use_units:
                for p in range(self.nspecies):
                    assert self[f'{p}_Coordinates'].unit == center.unit
            res = [func(f'{p}_Masses', f'{p}_Coordinates', f'{p}_Velocities', center)
                   for p in range(self.nspecies)]
            return np.sum(np.vstack(res), axis=0), res
        else:
            p = parttype
            if settings.use_units:
                assert self[f'{p}_Coordinates'].unit == center.unit
            return func(f'{p}_Masses', f'{p}_Coordinates', f'{p}_Velocities', center)
