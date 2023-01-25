from .arepo_catalog import Catalog
from .arepo_converter import ArepoConverter
import numpy as np
import os
import time
import h5py
from . import settings


class Snapshot(dict):
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

    def __init__(self, basedir, snapnum, basename="snap", verbose=False,
                 no_snapdir=False, load_catalog=None,
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

        self.basedir = basedir
        self.snapnum = snapnum
        self.basename = basename
        self.verbose = verbose
        self.no_snapdir = no_snapdir
        self.load_catalog = load_catalog

        self.dic_selection_index = dic_selection_index

        # in case single file
        self.snapname = self.basedir + "/" + \
            basename + "_" + str(self.snapnum).zfill(3)
        self.multi_file = False
        self.first_snapfile_name = self.snapname + ".hdf5"

        # if multiple files
        if not os.path.exists(self.first_snapfile_name):
            if not self.no_snapdir:
                self.snapname = self.basedir + "/" + "snapdir_" + \
                    str(self.snapnum).zfill(3) + "/" + \
                    basename + "_" + str(self.snapnum).zfill(3)
            else:
                self.snapname = self.basedir + "/" + \
                    basename + "_" + str(self.snapnum).zfill(3)
            self.first_snapfile_name = self.snapname+".0.hdf5"
            assert os.path.exists(self.first_snapfile_name)
            self.multi_file = True

        if self.verbose:
            print("snapshot", snapnum, "found")

        # get header of first file
        f = h5py.File(self.first_snapfile_name, 'r')

        self.Header = dict(f['Header'].attrs)
        self.Parameters = dict(f['Parameters'].attrs)
        self.Config = dict(f['Config'].attrs)

        f.close()

        self.converter = ArepoConverter(self.first_snapfile_name)
        if self.Parameters['ComovingIntegrationOn'] == 1:
            self.age = self.converter.age
            self.lookback_time = self.converter.lookback_time
            self.z = self.converter.z
            if self.verbose:
                print("at z =", self.z)
        else:
            self.time = self.converter.time

        self.nfiles = self.Header["NumFilesPerSnapshot"]
        self.npart = self.Header["NumPart_Total"]
        self.nspecies = self.npart.size

        if self.verbose:
            print("has", self.nspecies, "particle types")
            print("with npart =", self.npart)

        self.a = self.converter.a
        self.h = self.converter.h

        self.box = self.Header["BoxSize"]
        box_size = [self.box, self.box, self.box]

        for ii, dim in enumerate(['X', 'Y', 'Z']):
            if 'LONG_' + dim in self.Config:
                box_size[ii] *= self.Config['LONG_' + dim]

        if settings.use_units:
            get_paicos_quantity = self.converter.get_paicos_quantity
            self.box_size = get_paicos_quantity(box_size, 'Coordinates')
            self.masstable = get_paicos_quantity(self.Header["MassTable"],
                                                 'Masses')
        else:
            self.box_size = np.array(box_size)
            self.masstable = self.Header["MassTable"]

        # get subfind catalog?
        if load_catalog is None:
            if self.Parameters['ComovingIntegrationOn'] == 1:
                load_catalog = True
            else:
                load_catalog = False

        if load_catalog:
            try:
                self.Cat = Catalog(
                    self.basedir, self.snapnum, verbose=self.verbose,
                    subfind_catalog=True, converter=self.converter)
            except FileNotFoundError:
                self.Cat = None

            # If no subfind catalog found, then try for a fof catalog
            if self.Cat is None:
                try:
                    self.Cat = Catalog(
                        self.basedir, self.snapnum, verbose=self.verbose,
                        subfind_catalog=False, converter=self.converter)
                except FileNotFoundError:
                    import warnings
                    warnings.warn('no catalog found')

        self.P_attrs = dict()  # attributes

        if 'GAMMA' in self.Config:
            self.gamma = self.Config['GAMMA']
        elif 'ISOTHERMAL' in self.Config:
            self.gamma = 1
        else:
            self.gamma = 5/3

        self.derived_data_counter = 0

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
        with h5py.File(self.first_snapfile_name, 'r') as file:
            if PartType_str in list(file.keys()):
                keys = list(file[PartType_str].keys())
                if verbose:
                    print('\nKeys for ' + PartType_str + ' in the hdf5 file:')
                    for key in (sorted(keys)):
                        print(key)
                    # if PartType == 0:
                    from .derived_variables import get_variable_function
                    print('\nPossible derived variables are:')
                    keys = get_variable_function(str(PartType) + '_', True)
                    for key in (sorted(keys)):
                        print(key)
                    return None
                else:
                    return keys
            else:
                print('PartType not in hdf5 file')

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
        if blockname not in self.info(particle_type, False):
            msg = 'Unable to load parttype {}, blockname {} as this field is not in the hdf5 file'
            raise RuntimeError(msg.format(particle_type, blockname))

        datname = "PartType"+str(particle_type)+"/"+blockname
        PartType_str = 'PartType{}'.format(particle_type)
        if P_key in self:
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
            cur_filename = self.snapname

            if self.multi_file:
                cur_filename += "." + str(ifile)

            cur_filename += ".hdf5"

            f = h5py.File(cur_filename, "r")

            np_file = f["Header"].attrs["NumPart_ThisFile"][particle_type]

            if ifile == 0:   # initialize array
                if f[datname].shape.__len__() == 1:
                    self[P_key] = np.empty(
                        self.npart[particle_type], dtype=f[datname].dtype)
                else:
                    self[P_key] = np.empty(
                        (self.npart[particle_type], f[datname].shape[1]),
                        dtype=f[datname].dtype)
                # Load attributes
                data_attributes = dict(f[PartType_str][blockname].attrs)
                if len(data_attributes) > 0:
                    data_attributes.update({'small_h': self.h,
                                            'scale_factor': self.a})
                self.P_attrs[P_key] = data_attributes

            self[P_key][skip_part:skip_part+np_file] = f[datname]

            skip_part += np_file

        if settings.double_precision:
            # Load all variables with double precision
            self[P_key] = self[P_key].astype(np.float64)
        else:
            import warnings
            warnings.warn('\n\nThe cython routines expect double precision ' +
                          'and will fail unless settings.double_precision ' +
                          'is True.\n\n')

        # Only keep the cells with True in the selection index array
        if particle_type in self.dic_selection_index.keys():
            selection_index = self.dic_selection_index[particle_type]
            shape = self[P_key].shape
            if len(shape) == 1:
                self[P_key] = self[P_key][selection_index]
            elif len(shape) == 2:
                self[P_key] = self[P_key][selection_index, :]
            else:
                raise RuntimeError('Data has unexpected shape!')

        if settings.use_units or give_units:
            try:
                self[P_key] = self.converter.get_paicos_quantity(self[P_key],
                                                                 blockname)
            except:
                from warnings import warn
                warn('Failed to give {} units'.format(P_key))

        if self.verbose:
            print("... done! (took", time.time()-start_time, "s)")

    def get_derived_data(self, particle_type, blockname, verbose=False):
        """
        Get derived quantities. Example usage:

        snap = Snapshot(...)

        snap.get_derived_data(0, 'Temperatures')

        """
        from .derived_variables import get_variable_function

        P_key = str(particle_type) + "_" + blockname

        msg = ('\n\n{} is in the hdf5 file(s), please use load_data instead ' +
               'of get_derived_data').format(blockname)
        assert blockname not in self.info(particle_type, False), msg

        if verbose:
            msg1 = 'Attempting to get derived variable: {}...'.format(P_key)
            msg2 = 'So we need the variable: {}...'.format(P_key)
            if self.derived_data_counter == 0:
                print(msg1, end='')
            else:
                print('\n\t' + msg2, end='')
            self.derived_data_counter += 1

        func = get_variable_function(P_key)
        self[P_key] = func(self)

        if verbose:
            self.derived_data_counter -= 1
            if self.derived_data_counter == 0:
                print('\t[DONE]\n')

    def __getitem__(self, key):
        """
        This method is a special method in Python classes, known as a "magic
        method" that allows instances of the class to be accessed like a
        dictionary, using the bracket notation.

        This method is used to access the data stored in the class, it takes a
        single argument:

        key : a string that represents the data that is being accessed, it
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

        if key not in self.keys():
            if not key[0].isnumeric() or key[1] != '_':
                msg = ('\n\nKeys are expected to consist of an integer ' +
                       '(the particle type) and a blockname, separated by a ' +
                       ' _. For instance 0_Density. You can get the ' +
                       'available fields like so: snap.info(0)')
                raise RuntimeError(msg)
            parttype = int(key[0])
            name = key[2:]
            if name in self.info(parttype, False):
                self.load_data(parttype, name)
            else:
                verbose = settings.print_info_when_deriving_variables
                self.get_derived_data(parttype, name, verbose=verbose)

        return super().__getitem__(key)

    def remove_data(self, particle_type, blockname):
        """
        Remove data from object. Sometimes useful for for large datasets
        """
        P_key = str(particle_type)+"_"+blockname
        if P_key in self:
            del self[P_key]
        if P_key in self.P_attrs:
            del self.P_attrs[P_key]

    def get_volumes(self):
        self["0_Volume"]
        from warnings import warn
        warn(("This method will be soon deprecated in favor of automatic " +
              " loading using:\n\n" +
              " snap['0_Volume']\n\n or the explicit command\n\n" +
              "snap.get_derived_data(0, 'Volume')"),
             DeprecationWarning, stacklevel=2)

    def get_temperatures(self):
        self['0_Temperatures']
        from warnings import warn
        warn(("This method will be soon deprecated in favor of automatic " +
              " loading using:\n\n" +
              " snap['0_Temperatures']\n\n or the explicit command\n\n" +
              "snap.get_derived_data(0, 'Temperatures')"),
             DeprecationWarning, stacklevel=2)

    # find subhalos that particles belong to
    def get_host_subhalos(self, particle_type):
        if str(particle_type)+"_HostSub" in self.P:
            return

        if self.verbose:
            print("getting host subhalos ...")
            start_time = time.time()

        # -2 if not part of any FoF group
        hostsubhalos = -2*np.ones(self.npart[particle_type], dtype=np.int32)
        hostsubhalos[0:(self.Cat.Group["GroupLenType"][:, particle_type]).sum(
            dtype=np.int64)] = -1   # -1 if not part of any subhalo

        firstpart_gr = 0
        cur_sub = 0
        for igr in range(self.Cat.ngroups):
            firstpart_sub = firstpart_gr

            for isub in range(self.Cat.Group["GroupNsubs"][igr]):
                hostsubhalos[firstpart_sub:firstpart_sub +
                             self.Cat.Sub["SubhaloLenType"][cur_sub, particle_type]] = cur_sub

                firstpart_sub += self.Cat.Sub["SubhaloLenType"][cur_sub,
                                                                particle_type]
                cur_sub += 1

            firstpart_gr += self.Cat.Group["GroupLenType"][igr, particle_type]

        self.P[str(particle_type)+"_HostSub"] = hostsubhalos

        if self.verbose:
            print("... done! (took", time.time()-start_time, "s)")

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
                               no_snapdir=self.no_snapdir,
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
