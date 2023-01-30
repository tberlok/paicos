from .arepo_catalog import Catalog
from .paicos_readers import PaicosReader
import numpy as np
import time
import h5py
from . import settings


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

    Important methods and attributes.

    snap = Snapshot()

    snap.Parameters (dict): Contains information from the parameter
                            file used in the simulation (e.g. param.txt).

    snap.Config (dict): Contains information from the Config
                        file used in the simulation (e.g. Config.txt).

    snap.Header (dict): Contains information about this particular snapshot
                        such as its time (e.g scale factor).

    snap.z (float): redshift

    snap.h (float): reduced Hubble param (e.g. 0.67)

    snap.age: the age of the Universe (only for cosmological runs)
    snap.lookback_time: the age of the Universe (only for cosmological runs)

    snap.time: the time stamp of the snapshot (only for non-cosmological runs)

    snap.box_size: the dimensions of the simulation domain

    """

    def __init__(self, basedir, snapnum, basename="snap", load_all=False,
                 to_physical=False, load_catalog=None, verbose=False,
                 dic_selection_index={}):
        """
        Initialize the Snapshot class.

        Parameters:

        basedir (str): path of the directory containing the snapshot
                       (e.g. the 'output' folder)

        snapnum (int): snapshot number

        basename (str): name of the snapshot file, default is "snap"

        verbose (bool): whether to print information about the snapshot,
        default is False

        no_snapdir (bool): whether there is no snap directory, i.e.,
                           default is False

        load_catalog (bool): whether to load the subfind catalog.
                             The default None is internally changed to
                             True for comoving simulations and to
                             False for non-comoving simulations.
    """

        super().__init__(basedir=basedir, snapnum=snapnum, basename=basename,
                         load_all=load_all, to_physical=to_physical,
                         basesubdir='snapdir', verbose=verbose)

        self.load_catalog = load_catalog

        self.dic_selection_index = dic_selection_index

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
            if self.ComovingIntegrationOn:
                load_catalog = True
            else:
                load_catalog = False

        if load_catalog:
            try:
                self.Cat = Catalog(
                    self.basedir, self.snapnum, verbose=self.verbose,
                    subfind_catalog=True)
            except FileNotFoundError:
                self.Cat = None

            # If no subfind catalog found, then try for a fof catalog
            if self.Cat is None:
                try:
                    self.Cat = Catalog(
                        self.basedir, self.snapnum, verbose=self.verbose,
                        subfind_catalog=False)
                except FileNotFoundError:
                    import warnings
                    warnings.warn('no catalog found')

        self.P_attrs = dict()  # attributes

        self.derived_data_counter = 0

        self._add_mass_to_user_funcs()

        self._find_available_for_loading()
        self._find_available_functions()

        self.__get_auto_comple_list()

    def _add_mass_to_user_funcs(self):

        self._this_snap_funcs = {}

        class Mass:
            def __init__(self, parttype):
                self.parttype = parttype

            def get_masses_from_header(self, snap):
                """
                Get mass of particle type from the mass table
                """
                parttype = self.parttype
                if parttype in snap.dic_selection_index.keys():
                    npart = snap.dic_selection_index[parttype].shape[0]
                else:
                    npart = snap.npart[parttype]
                return np.ones(npart)*snap.masstable[parttype]

        for parttype in range(self.nspecies):
            if self.masstable[parttype] != 0:
                P_key = str(parttype) + '_Masses'
                obj = Mass(parttype)
                self._this_snap_funcs[P_key] = obj.get_masses_from_header

    def _find_available_for_loading(self):
        self._all_avail_load = []
        for PartType in range(self.nspecies):
            PartType_str = 'PartType{}'.format(PartType)
            with h5py.File(self.filename, 'r') as file:
                if PartType_str in list(file.keys()):
                    load_keys = list(file[PartType_str].keys())
                    for key in load_keys:
                        P_key = str(PartType) + '_' + key
                        self._all_avail_load.append(P_key)

    def _find_available_functions(self):
        from .settings import use_only_user_functions
        from . import derived_variables
        from inspect import signature

        user_functs = derived_variables.user_functions

        for key in user_functs.keys():
            self._this_snap_funcs.update({key: user_functs[key]})

        if not use_only_user_functions:
            def_functs = derived_variables.default_functions
            for key in def_functs.keys():
                if key not in self._this_snap_funcs.keys():
                    self._this_snap_funcs.update({key: def_functs[key]})

        self._dependency_dic = {}
        for key in self._this_snap_funcs.keys():
            func = self._this_snap_funcs[key]
            sig = signature(func)
            if len(sig.parameters) == 2:
                self._dependency_dic[key] = func(self, True)
            else:
                self._dependency_dic[key] = []

        dependency_dic = dict(self._dependency_dic)

        # This will fail for very nested dependencies.
        for jj in range(3):
            # First substitute all dependencies that are
            # at the top level of the dictionary
            for key in dependency_dic.keys():
                deps = dependency_dic[key]
                for dep in list(deps):
                    if dep in dependency_dic.keys():
                        deps.remove(dep)
                        for subdep in dependency_dic[dep]:
                            deps.append(subdep)

            # Then remove all the dependencies that can be loaded
            for key in dependency_dic.keys():
                deps = dependency_dic[key]
                for dep in list(deps):
                    if dep in self._all_avail_load:
                        deps.remove(dep)

        # Delete the entries where we do not have the requirements
        for key in dependency_dic.keys():
            dep = len(dependency_dic[key])
            if dep > 0:
                if key in user_functs.keys():
                    import warnings
                    msg = ('Deleting the user function: {} because its ' +
                           'dependency: {} is missing')
                    warnings.warn(msg.format(user_functs[key],
                                  dependency_dic[key]))
                del self._this_snap_funcs[key]

    def get_variable_function(self, P_key, info=False):

        assert type(P_key) is str

        if not P_key[0].isnumeric() or P_key[1] != '_':
            msg = ('\n\nKeys are expected to consist of an integer ' +
                   '(the particle type) and a blockname, separated by a ' +
                   ' _. For instance 0_Density. You can get the ' +
                   'available fields like so: snap.info(0)')
            raise RuntimeError(msg)

        if not info:
            if P_key in self._this_snap_funcs.keys():
                return self._this_snap_funcs[P_key]
            else:
                msg = '\n\n{} not found in the functions: {}'
                msg = msg.format(P_key, self._this_snap_funcs)
                raise RuntimeError(msg)

        # Return a list with all available keys for this parttype
        if info:
            parttype = int(P_key[0])

            avail_list = []
            for key in self._this_snap_funcs.keys():
                if int(key[0]) == parttype:
                    avail_list.append(key)

            return avail_list

    def info(self, PartType, verbose=True):
        """
        This function provides information about the keys of a certain
        particle type in a snapshot file.

        Args: PartType (int): An integer representing the particle type of
        interest. verbose (bool, optional): A flag indicating whether or not
        to print the keys to the console. Defaults to True.

        Returns: list or None : If the PartType exists in the file, a list of
        keys for that PartType is returned, otherwise None.

        This function opens the snapshot file and checks if the PartType
        passed as an argument exists in the file. If it does, it retrieves the
        keys of that PartType and if verbose is True, prints the keys to the
        console, otherwise it returns the keys. If the PartType does not exist
        in the file, the function will print "PartType not in hdf5 file" to
        the console.

        The function can be useful for examining the contents of a snapshot
        file and determining what data is available for a given particle
        type.
        """
        PartType_str = 'PartType{}'.format(PartType)
        with h5py.File(self.filename, 'r') as file:
            if PartType_str in list(file.keys()):
                load_keys = list(file[PartType_str].keys())
                load_keys = [str(PartType) + '_' + key for key in load_keys]
                if verbose:
                    print('\nKeys for ' + PartType_str + ' in the hdf5 file:')
                    for key in (sorted(load_keys)):
                        if settings.use_aliases:
                            if key in settings.aliases.keys():
                                alias = settings.aliases[key]
                                msg = alias + '\t'*5 + '(an alias of {})'
                                print(msg.format(key))
                            else:
                                print(key)
                        else:
                            print(key)

                    print('\nPossible derived variables are:')
                    dkeys = self.get_variable_function(str(PartType) + '_', True)

                    for key in (sorted(dkeys)):
                        if settings.use_aliases:
                            if key in settings.aliases.keys():
                                alias = settings.aliases[key]
                                msg = alias + '\t'*5 + '(an alias of {})'
                                print(msg.format(key))
                            else:
                                print(key)
                        else:
                            print(key)
                    return None
                else:
                    return load_keys
            else:
                if verbose:
                    print('PartType not in hdf5 file')
                else:
                    return []

    def load_data(self, particle_type, blockname, give_units=False):
        """
        Load data from hdf5 file(s). Example usage:

        snap = Snapshot(...)

        snap.load_data(0, 'Density')

        Note that subsequent calls does not reload the data. Reloading
        the data can be done explicitly:

        snap.remove_data(0, 'Density')
        snap.load_data(0, 'Density')

        """

        assert particle_type < self.nspecies

        P_key = str(particle_type)+"_"+blockname
        alias_key = P_key

        if settings.use_aliases:
            if P_key in settings.aliases.keys():
                alias_key = settings.aliases[P_key]

        if P_key not in self.info(particle_type, False):
            msg = 'Unable to load parttype {}, blockname {} as this field is not in the hdf5 file'
            raise RuntimeError(msg.format(particle_type, blockname))

        datname = "PartType"+str(particle_type)+"/"+blockname
        PartType_str = 'PartType{}'.format(particle_type)
        if alias_key in self:
            if self.verbose:
                print(blockname, "for species",
                      particle_type, "already in memory")
            return
        elif self.verbose:
            print("loading block", blockname,
                  "for species", particle_type, "...")
            start_time = time.time()

        skip_part = 0

        for ifile in range(self.nfiles):
            if self.multi_file is False:
                cur_filename = self.filename
            else:
                if self.no_subdir:
                    cur_filename = self.multi_wo_dir.format(ifile)
                else:
                    cur_filename = self.multi_file.format(ifile)

            f = h5py.File(cur_filename, "r")

            np_file = f["Header"].attrs["NumPart_ThisFile"][particle_type]

            if ifile == 0:   # initialize array
                if f[datname].shape.__len__() == 1:
                    self[alias_key] = np.empty(
                        self.npart[particle_type], dtype=f[datname].dtype)
                else:
                    self[alias_key] = np.empty(
                        (self.npart[particle_type], f[datname].shape[1]),
                        dtype=f[datname].dtype)
                # Load attributes
                data_attributes = dict(f[PartType_str][blockname].attrs)

                self.P_attrs[alias_key] = data_attributes

            self[alias_key][skip_part:skip_part+np_file] = f[datname]

            skip_part += np_file

        if settings.double_precision:
            # Load all variables with double precision
            import numbers
            if not issubclass(self[alias_key].dtype.type, numbers.Integral):
                self[alias_key] = self[alias_key].astype(np.float64)
        else:
            import warnings
            warnings.warn('\n\nThe cython routines expect double precision ' +
                          'and will fail unless settings.double_precision ' +
                          'is True.\n\n')

        # Only keep the cells with True in the selection index array
        if particle_type in self.dic_selection_index.keys():
            selection_index = self.dic_selection_index[particle_type]
            shape = self[alias_key].shape
            if len(shape) == 1:
                self[alias_key] = self[alias_key][selection_index]
            elif len(shape) == 2:
                self[alias_key] = self[alias_key][selection_index, :]
            else:
                raise RuntimeError('Data has unexpected shape!')

        if settings.use_units or give_units:
            try:
                self[alias_key] = self.get_paicos_quantity(self[alias_key],
                                                           blockname)
            except:
                from warnings import warn
                warn('Failed to give {} units'.format(alias_key))

        if self.verbose:
            print("... done! (took", time.time()-start_time, "s)")

    def get_derived_data(self, particle_type, blockname, verbose=False):
        """
        Get derived quantities. Example usage:

        snap = Snapshot(...)

        snap.get_derived_data(0, 'Temperatures')

        """
        # from .derived_variables import get_variable_function

        P_key = str(particle_type) + "_" + blockname

        msg = ('\n\n{} is in the hdf5 file(s), please use load_data instead ' +
               'of get_derived_data').format(blockname)
        assert P_key not in self.info(particle_type, False), msg

        if verbose:
            msg1 = 'Attempting to get derived variable: {}...'.format(P_key)
            msg2 = 'So we need the variable: {}...'.format(P_key)
            if self.derived_data_counter == 0:
                print(msg1, end='')
            else:
                print('\n\t' + msg2, end='')
            self.derived_data_counter += 1

        func = self.get_variable_function(P_key)

        if settings.use_aliases:
            if P_key in settings.aliases.keys():
                P_key = settings.aliases[P_key]
        self[P_key] = func(self)

        if verbose:
            self.derived_data_counter -= 1
            if self.derived_data_counter == 0:
                print('\t[DONE]\n')

    def __getitem__(self, P_key):
        """
        This method is a special method in Python classes, known as a "magic
        method" that allows instances of the class to be accessed like a
        dictionary, using the bracket notation.

        This method is used to access the data stored in the class, it takes a
        single argument:

        P_key : a string that represents the data that is being accessed, it
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

            if P_key in settings.inverse_aliases.keys():
                P_key = settings.inverse_aliases[P_key]

        if P_key not in self.keys():
            if not P_key[0].isnumeric() or P_key[1] != '_':
                msg = ('\n\nKeys are expected to consist of an integer ' +
                       '(the particle type) and a blockname, separated by a ' +
                       ' _. For instance 0_Density. You can get the ' +
                       'available fields like so: snap.info(0)')
                raise RuntimeError(msg)
            parttype = int(P_key[0])
            name = P_key[2:]

            if parttype >= self.nspecies:
                msg = 'Simulation only has {} species.'
                raise RuntimeError(msg.format(self.nspecies))

            if P_key in self.info(parttype, False):
                self.load_data(parttype, name)
            else:
                verbose = settings.print_info_when_deriving_variables
                self.get_derived_data(parttype, name, verbose=verbose)

        if settings.use_aliases:
            if P_key in settings.aliases.keys():
                P_key = settings.aliases[P_key]
        return super().__getitem__(P_key)

    def __get_auto_comple_list(self):
        self._auto_list = []

        self._auto_list = self._all_avail_load
        for key in self._this_snap_funcs.keys():
            self._auto_list.append(key)

        if settings.use_aliases:
            for ii, P_key in enumerate(self._auto_list):
                if P_key in settings.aliases.keys():
                    self._auto_list[ii] = settings.aliases[P_key]

    def _ipython_key_completions_(self):
        """
        Auto-completion of dictionary.

        Only works with variables that can directly loaded.
        """

        return self._auto_list

    def remove_data(self, particle_type, blockname):
        """
        Remove data from object. Sometimes useful for for large datasets
        """
        P_key = str(particle_type)+"_"+blockname
        if P_key in self:
            del self[P_key]
        if P_key in self.P_attrs:
            del self.P_attrs[P_key]

    def select(self, selection_index, parttype=0):
        """
        Create a new snapshot object which will only contain
        cells with a selection_index.

        Example use:
        index = snap['0_Density'] > snap['0_Density'].unit_quantity*1e-6
        selected_snap = snap.select(index)

        """

        s_index = selection_index

        if parttype in self.dic_selection_index.keys():
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
                               load_catalog=self.load_catalog,
                               dic_selection_index=dic_selection_index)

        for key in self.keys():
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

    def save_new_snapshot(self, basename, single_precision=False):
        """
        Save a new snapshot containing the currently loaded (derived)
        variables. Useful for reducing datasets to smaller sizes.
        """
        from .paicos_writer import PaicosWriter
        import h5py

        writer = PaicosWriter(self, self.basedir, basename, 'w')

        new_npart = [0]*self.nspecies
        for key in self.keys():
            for parttype in range(self.nspecies):
                if key[:2] == '{}_'.format(parttype):
                    new_npart[parttype] = self[key].shape[0]
                    PartType_str = 'PartType{}'.format(parttype)

                    if single_precision:
                        data = self[key].astype(np.float32)
                    else:
                        data = self[key]
                    writer.write_data(key[2:], data, group=PartType_str)

        with h5py.File(writer.tmp_filename, 'r+') as f:
            f['Header'].attrs["NumFilesPerSnapshot"] = 1
            f['Header'].attrs["NumPart_Total"] = np.array(new_npart)

        writer.finalize()
