"""
Defines the PaicosReader, Histogram2DReader and ImageReader which
can be used to load derived variables.
"""
import os
import h5py
import numpy as np
import traceback
from astropy import units as u
from astropy.cosmology import LambdaCDM
from .. import util
from .. import units as pu
from .. import settings
from ..orientation import Orientation
from ..image_creators.image_creator import ImageCreator


class PaicosReader(dict):
    """
    The PaicosReader can read any hdf5 file that contains the three
    groups: Header, Config and Parameters.

    It uses these to automatically construct a variety quantities
    which become accessible as properties.

    This class is subclassed by the Snapshot and Catalog classes.
    """

    def __init__(self, basedir='.', snapnum=None, basename="snap",
                 basesubdir='snapdir', load_all=True, to_physical=False,
                 verbose=False):

        """
        Initialize the PaicosReader class.

        Parameters
        ----------

            basedir : str
                The path of the directory containing the hdf5 files
                (e.g. the 'output' folder).

            snapnum : int
                e.g. the snapshot number

            basename : str
                name of the file takes the form ``basename_{:03d}.hdf5``
                or ``basename_{:03d}.{}.hdf5``. Default is ``snap``.

            basesubdir : str
                The name of the subfolder. Default is ``snapdir``.

            load_all : bool
                Whether to simply load all data, default is True.

            to_physical : bool
                Whether to convert from comoving to physical
                variables upon load.

            verbose : bool
                Whether to print information, default is False.
        """

        self.to_physical = to_physical
        if to_physical and not settings.use_units:
            err_msg = "to_physical=True requires that units are enabled"
            raise RuntimeError(err_msg)

        self.verbose = verbose

        if '.hdf5' in basedir:
            # User is trying to load a single hdf5 file directly, let's see if
            # that can work
            self.filename = str(basedir)
            self.multi_file = False
            basedir, basename, snapnum = util._split_filename(self.filename)
            self.basedir = basedir
            self.snapnum = snapnum
            self.basename = basename
        else:
            self.basedir = basedir
            self.snapnum = snapnum
            self.basename = basename

            self.basesubdir = basesubdir

            # Add a forward slash at the end of basedir if not already present
            if basedir[-1] != '/':
                basedir += '/'

            # If snapnum is None, filename is basedir + basename + '.hdf5'
            if snapnum is None:
                self.filename = basedir + basename + '.hdf5'
                err_msg = f'File: {self.filename} not found'
                if not os.path.exists(self.filename):
                    raise FileNotFoundError(err_msg)
            else:
                # Set filenames for single_file and multi_file cases
                single_file = basename + "_{:03d}.hdf5"
                multi_file = basesubdir + '_{:03d}/' + basename + '_{:03d}.{}.hdf5'
                multi_wo_dir = basename + '_{:03d}.{}.hdf5'

                single_file = basedir + single_file.format(snapnum)
                multi_file = basedir + multi_file.format(snapnum, snapnum, '{}')
                multi_wo_dir = basedir + multi_wo_dir.format(snapnum, '{}')

                # Check if single_file exists
                if os.path.exists(single_file):
                    self.multi_file = False
                    self.filename = single_file
                elif os.path.exists(multi_file.format(0)):
                    self.multi_file = True
                    self.first_file_name = self.filename = multi_file.format(0)
                    self.multi_filename = multi_file
                    self.no_subdir = False
                elif os.path.exists(multi_wo_dir.format(0)):
                    self.multi_file = True
                    self.first_file_name = self.filename = multi_wo_dir.format(0)
                    self.multi_filename = multi_file
                    self.no_subdir = True
                else:
                    err_msg = "File not found. Tried locations:\n{}\n{}\n{}"
                    err_msg = err_msg.format(single_file, multi_file.format(0),
                                             multi_wo_dir.format(0))
                    raise FileNotFoundError(err_msg)

        with h5py.File(self.filename, 'r') as f:
            self.Header = dict(f['Header'].attrs)
            self.Config = dict(f['Config'].attrs)
            self.Parameters = dict(f['Parameters'].attrs)
            keys = list(f.keys())

        # Enable units
            self.get_units_and_other_parameters()
        if settings.use_units:
            self.enable_units()
            self.add_user_units()

        # Find the adiabatic index
        self.gamma = self.get_adiabiatic_index()

        # Load all data sets
        if load_all:
            for key in keys:
                self.load_data(key)

        self.load_org_info()

    def load_org_info(self):
        """
        Load some extra info about original data.

        :meta private:
        """
        with h5py.File(self.filename, 'r') as f:
            if 'org_info' in f:
                self['org_info'] = {}
                for key in f['org_info'].attrs.keys():
                    self['org_info'][key] = f['org_info'].attrs[key]

    def get_units_and_other_parameters(self):
        """
        Define arepo units, scale factor (a) and h (HubbleParam).

        The input required is a hdf5 file with the Parameters and Header
        groups found in arepo snapshots.

        :meta private:
        """
        self._Time = self.Header['Time']
        self._HubbleParam = self.Parameters['HubbleParam']

        ComovingIntegrationOn = self.Parameters['ComovingIntegrationOn']

        self.ComovingIntegrationOn = bool(ComovingIntegrationOn)
        self.comoving_sim = self.ComovingIntegrationOn

        if self.ComovingIntegrationOn:
            self._Redshift = self.Header['Redshift']

            Omega0 = self.Parameters['Omega0']
            OmegaBaryon = self.Parameters['OmegaBaryon']
            OmegaLambda = self.Parameters['OmegaLambda']

            # Set up LambdaCDM cosmology to calculate times, etc
            self.cosmo = LambdaCDM(H0=100 * self.h, Om0=Omega0,
                                   Ob0=OmegaBaryon, Ode0=OmegaLambda)
            # Current age of the universe and look back time
            self._age = self.get_age(self.z)
            self._lookback_time = self.get_lookback_time(self.z)

        missing_unitinfo = True
        if 'UnitLength_in_cm' in self.Parameters:
            if 'UnitMass_in_g' in self.Parameters:
                if 'UnitVelocity_in_cm_per_s' in self.Parameters:
                    missing_unitinfo = False

        if settings.use_units and missing_unitinfo:
            raise RuntimeError("Units mssing in self.Parameters!")

        if not missing_unitinfo:
            unit_length = self.Parameters['UnitLength_in_cm'] * u.cm
            unit_mass = self.Parameters['UnitMass_in_g'] * u.g
            unit_velocity = self.Parameters['UnitVelocity_in_cm_per_s'] * u.cm / u.s
            unit_time = unit_length / unit_velocity
            unit_energy = unit_mass * unit_velocity**2
            unit_pressure = (unit_mass / unit_length) / unit_time**2
            unit_density = unit_mass / unit_length**3

            self.arepo_units_in_cgs = {'unit_length': unit_length,
                                       'unit_mass': unit_mass,
                                       'unit_velocity': unit_velocity,
                                       'unit_time': unit_time,
                                       'unit_energy': unit_energy,
                                       'unit_pressure': unit_pressure,
                                       'unit_density': unit_density}

        if settings.use_units:

            _ns = globals()

            try:
                arepo_mass = u.def_unit(
                    ["arepo_mass"],
                    unit_mass,
                    prefixes=False,
                    namespace=_ns,
                    doc="Arepo mass unit",
                    format={"latex": r"arepo\_mass"},
                )
                arepo_time = u.def_unit(
                    ["arepo_time"],
                    unit_time,
                    prefixes=False,
                    namespace=_ns,
                    doc="Arepo time unit",
                    format={"latex": r"arepo\_time"},
                )
                arepo_length = u.def_unit(
                    ["arepo_length"],
                    unit_length,
                    prefixes=False,
                    namespace=_ns,
                    doc="Arepo length unit",
                    format={"latex": r"arepo\_length"},
                )
                arepo_velocity = u.def_unit(
                    ["arepo_velocity"],
                    unit_velocity,
                    prefixes=False,
                    namespace=_ns,
                    doc="Arepo velocity unit",
                    format={"latex": r"arepo\_velocity"},
                )
                arepo_pressure = u.def_unit(
                    ["arepo_pressure"],
                    unit_pressure,
                    prefixes=False,
                    namespace=_ns,
                    doc="Arepo pressure unit",
                    format={"latex": r"arepo\_pressure"},
                )
                arepo_energy = u.def_unit(
                    ["arepo_energy"],
                    unit_energy,
                    prefixes=False,
                    namespace=_ns,
                    doc="Arepo energy unit",
                    format={"latex": r"arepo\_energy"},
                )
                arepo_density = u.def_unit(
                    ["arepo_density"],
                    unit_density,
                    prefixes=False,
                    namespace=_ns,
                    doc="Arepo density unit",
                    format={"latex": r"arepo\_density"},
                )
            except ValueError:
                traceback.print_exc()
                err_msg = ("Re-declaration of arepo code units attempted "
                           + "within same Python session. "
                           + "See https://github.com/tberlok/paicos/issues/59")
                raise RuntimeError(err_msg)

            self.arepo_units = {'unit_length': arepo_length,
                                'unit_mass': arepo_mass,
                                'unit_velocity': arepo_velocity,
                                'unit_time': arepo_time,
                                'unit_energy': arepo_energy,
                                'unit_pressure': arepo_pressure,
                                'unit_density': arepo_density}

    def enable_units(self):
        """
        Enables arepo units globally.

        :meta private:
        """
        for unit_name, unit in self.arepo_units.items():
            u.add_enabled_units(unit)
            phys_type = unit_name.split('_')[1]
            u.def_physical_type(unit, phys_type)

        self._length = self.get_paicos_quantity(1, 'Coordinates')
        self._mass = self.get_paicos_quantity(1, 'Masses')
        self._velocity = self.get_paicos_quantity(1, 'Velocities')

    @property
    def length(self):
        """The unit of length used in the simulation."""
        return self._length

    @property
    def mass(self):
        """The unit of mass used in the simulation."""
        return self._mass

    @property
    def velocity(self):
        """
        One of the units used for velocities in the simulation.

        Note: The a and h-scalings are not the same for velocities
        in the halo catalogs and in the snapshots.
        """
        return self._velocity

    def add_user_units(self):
        """
        Add all user supplied units

        :meta private:
        """
        # pylint: disable=import-outside-toplevel, consider-using-dict-items
        from .. import unit_specifications

        unit_dict = unit_specifications.unit_dict
        user_unit_dict = util.user_unit_dict

        err_msg = ("\n\nThe user supplied unit for '{}:{}' already exists "
                   + "in the default Paicos settings. Changing from '{}' to  "
                   + "'{}' is not allowed. Please make a pull request if you "
                   + " have found a bug.")

        for field in user_unit_dict:
            for name in user_unit_dict[field]:

                # Check for overwriting and changing of defaults
                if name in unit_dict[field]:
                    def_unit = unit_dict[field][name]
                    user_unit = user_unit_dict[field][name]
                    if user_unit != def_unit:
                        msg = err_msg.format(field, name, def_unit, user_unit)
                        del user_unit_dict[field][name]
                        raise RuntimeError(msg)

                # Set the new units
                unit_dict[field][name] = user_unit_dict[field][name]

    def get_adiabiatic_index(self):
        """
        Returns the adiabatic index
        """
        if 'GAMMA' in self.Config:
            gamma = self.Config['GAMMA']
        elif 'ISOTHERMAL' in self.Config:
            gamma = 1
        else:
            gamma = 5 / 3
        return gamma

    @property
    def a(self):
        """
        The scale factor.
        """
        if self.comoving_sim:
            return self._Time
        raise RuntimeError('Non-comoving object has no scale factor')

    @property
    def h(self):
        """
        The reduced Hubble parameter
        """
        if self._HubbleParam == 0:
            return 1.0
        return self._HubbleParam

    @property
    def z(self):
        """
        The redshift.
        """
        if self.comoving_sim:
            return self._Redshift
        raise RuntimeError('Non-comoving object has no redshift')

    @property
    def lookback_time(self):
        """ The lookback time. """
        if self.comoving_sim:
            return self._lookback_time
        raise RuntimeError('Non-comoving object has no lookback_time')

    @property
    def age(self):
        """ The age of the universe in the simulation. """
        if self.comoving_sim:
            return self._age
        raise RuntimeError('Non-comoving object has no lookback_time')

    @property
    def time(self):
        """
        The time elapsed since the beginning of the simulation.

        Only defined for non-comoving simulations.
        """
        if self.comoving_sim:
            msg = 'time not defined for comoving sim'
            raise RuntimeError(msg)

        time = self._Time * self.arepo_units['unit_time']
        if self._HubbleParam != 1.0:
            time = self._Time * self.unit_quantity('arepo_time / small_h')
        return time

    def rho_crit(self, z):
        """
        Returns the physical critical density (no a or h factors)
        """
        rho_crit = self.cosmo.critical_density(z).to('arepo_density')
        return self.__convert_to_paicos(rho_crit, z)

    def get_lookback_time(self, z):
        """
        Returns the lookback time for a given redshift, z.
        """
        lookback_time = self.cosmo.lookback_time(z)
        return self.__convert_to_paicos(lookback_time, z)

    def get_age(self, z):
        """
        Returns the age of the universe for a given redshift, z.
        """
        age = self.cosmo.lookback_time(1e100) - self.cosmo.lookback_time(z)

        return self.__convert_to_paicos(age, z)

    def __convert_to_paicos(self, time, z):
        """
        Helper function to convert astropy quantities in to PaicosQuantities.

        Really just to still have access to the .label() method.
        """
        if settings.use_units:
            a = 1.0 / (z + 1.)
            if isinstance(a, np.ndarray):
                time = pu.PaicosTimeSeries(time, a=a, h=self.h, copy=True,
                                           comoving_sim=self.comoving_sim)
            else:
                time = pu.PaicosQuantity(time, a=a, h=self.h, copy=True,
                                         comoving_sim=self.comoving_sim)
        return time

    def get_paicos_quantity(self, data, name, field='default'):
        """
        Convert some data to a PaicosQuantity.

        Parameters:

            data: typically some numpy array, integer or float

            name: corresponds to a block name in Arepo snapshots,
                  i.e., a key in one of the dictionaries defined
                  in unit_specifications.py

            field: the name of of one of the dictionaries defined in the
                   unit specifications, e.g,:
                    ['default', 'voronoi_cells', 'dark_matter',
                    'stars', 'black_holes', 'groups', 'subhalos']

        Returns: A PaicosQuantity
        """

        if hasattr(data, 'unit'):
            msg = f'Data already had units! {name}'
            raise RuntimeError(msg)

        unit = self.find_unit(name, field)

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if unit:
            return pu.PaicosQuantity(data, unit, a=self._Time, h=self.h,
                                     comoving_sim=self.comoving_sim,
                                     dtype=data.dtype)
        return data

    def find_unit(self, name, field):
        """
        Find unit for a given quantity.

        Parameters:

            name: corresponds to a block name in Arepo snapshots,
                  i.e., a key in one of the dictionaries defined
                  in unit_specifications.py

            field: the name of of one of the dictionaries defined in the
                   unit specifications, e.g,:
                    ['default', 'voronoi_cells', 'dark_matter',
                    'stars', 'black_holes', 'groups', 'subhalos']
        """
        # pylint: disable=import-outside-toplevel

        # This import statement can only be done after arepo units have
        # been globally enabled
        from .. import unit_specifications

        if field not in unit_specifications.unit_dict:
            raise RuntimeError(f'unknown field: {field}')

        if name not in unit_specifications.unit_dict[field]:
            unit = False
        else:
            unit = unit_specifications.unit_dict[field][name]

        # Convert string to astropy unit
        if isinstance(unit, str):
            unit = u.Unit(unit)

        if unit is False:
            msg = ('\n\nUnit for field:{}, blockname:{} not implemented!'
                   + '\nYou can get rid of this error by changing your '
                   + 'settings, i.e.,\n\npa.settings.strict_units = False\n\n'
                   + 'However, this means that this quantity will not be read in. '
                   + 'If you need this quantity, then the best way forward is to '
                   + 'add it to the unit_specifications '
                   + 'by using the pa.add_user_units '
                   + 'function. You can also add the pa.add_user_units call to your '
                   + 'Paicos user settings to avoid having to do this more than once. '
                   + 'Alternatively, please create an issue on GitHub if you think '
                   + 'others could benefit from that.'
                   + '\nFurther instructions can be '
                   + 'found by runnning pa.add_user_units? '
                   + 'in a terminal or notebook.')

            if settings.strict_units:
                raise RuntimeError(msg.format(field, name))
            return False

        return self._sanitize_unit(unit)

    def unit_quantity(self, astropy_unit_str):
        """
        Returns a Paicos quantity with value 1 and any
        astropy unit.
        """
        unit = u.Unit(astropy_unit_str)
        return pu.PaicosQuantity(1.0, unit, a=self._Time, h=self.h,
                                 comoving_sim=self.comoving_sim,
                                 dtype=float, copy=True)

    def uq(self, astropy_unit_str):
        """
        A short hand for the unit_quantity method.
        """
        return self.unit_quantity(astropy_unit_str)

    def _sanitize_unit(self, unit):
        """
        Removes 'a' factors for non-comoving simulations,
        and 'h' factors for simulations with HubbleParam=1.
        """

        remove_list = []
        if not self.ComovingIntegrationOn:
            remove_list.append(u.Unit('small_a'))

        if self.h == 1.:
            remove_list.append(u.Unit('small_h'))

        return pu.get_new_unit(unit, remove_list)

    def _load_helper(self, hdf5file, name, group=None):
        """
        Implementation of nested group functionality for readers
        """

        if group is None:
            path = name
        else:
            path = group + '/' + name

        terms = path.split('/')

        # If dataset, read it and place it in the right dictionary
        if isinstance(hdf5file[path], h5py.Dataset):
            data = util.load_dataset(hdf5file, name, group=group)

            # Convert to physical
            if isinstance(data, pu.PaicosQuantity) and self.to_physical:
                data = data.to_physical

            mydic = self
            if '/' not in path:
                mydic[name] = data
            else:
                for ii in range(len(terms) - 1):
                    mydic = mydic[terms[ii]]
                mydic.update({name: data})

        # If group, create top-level dictionary and call
        # recursively for every key in this group.
        elif isinstance(hdf5file[path], h5py.Group):
            upper_dic = self
            for ii in range(len(terms)):
                upper_key = terms[ii]
                if upper_key not in upper_dic:
                    upper_dic[upper_key] = {}
                upper_dic = upper_dic[upper_key]

            for subname in hdf5file[path].keys():
                self._load_helper(hdf5file, subname, group=path)

    def load_data(self, name, group=None):
        """
        Load data from a generic Paicos hdf5 file (written by a PaicosWriter
        instance).

        The method requires that the data sets have a 'unit' attribute
        and hence does not work for Arepo hdf5 files.
        For this reason, this method is overloaded in the Snapshot and Catalog
        classes.
        """
        with h5py.File(self.filename, 'r') as f:
            self._load_helper(f, name, group)

        for key in list(self.keys()):
            if isinstance(self[key], dict) and len(self[key].keys()) == 0:
                del self[key]

    def _load_data_old(self, name, group=None):
        """
        Load data from a generic Paicos hdf5 file (written by a PaicosWriter
        instance).

        The method requires that the data sets have a 'unit' attribute
        and hence does not work for Arepo hdf5 files.
        For this reason, this method is overloaded in the Snapshot and Catalog
        classes.
        """

        with h5py.File(self.filename, 'r') as f:
            if isinstance(f[name], h5py.Dataset):
                self[name] = util.load_dataset(f, name, group=group)
                if isinstance(self[name], pu.PaicosQuantity) and self.to_physical:
                    self[name] = self[name].to_physical
            elif isinstance(f[name], h5py.Group):
                for data_name in f[name].keys():
                    if isinstance(f[name][data_name], h5py.Group):
                        import warnings
                        warnings.warn("load_data: nested subgroups don't work :(")
                        # raise RuntimeError()
                    else:
                        data = util.load_dataset(f, data_name, group=name)
                        if name not in self:
                            self[name] = {}
                        if isinstance(data, pu.PaicosQuantity) and self.to_physical:
                            self[name][data_name] = data.to_physical
                        else:
                            self[name][data_name] = data


class ImageReader(PaicosReader):
    """
    This is a subclass of the PaicosReader.

    It reads the additional information stored in image files and makes
    them accessible as attributes, i.e., extent, widths, center, direction
    and image_creator.

    For projection files, it also tries to automatically get derived
    variables, e.g., if the hdf5 file contains

    'MagneticFieldSquaredTimesVolume' and 'Volume'

    then it will automatically divide them to obtain the MagneticFieldSquared.
    """

    def __init__(self, basedir='.', snapnum=None, basename="projection", load_all=True):
        """
        See documentation for the PaicosReader.

        Returns a dictionary with additional attributes.
        """

        # The PaicosReader class takes care of most of the loading
        super().__init__(basedir, snapnum, basename=basename,
                         load_all=load_all)

        # Load info specific to images
        with h5py.File(self.filename, 'r') as f:
            self.extent = util.load_dataset(f, 'extent', group='image_info')
            self.widths = util.load_dataset(f, 'widths', group='image_info')
            self.center = util.load_dataset(f, 'center', group='image_info')
            self.direction = direction = f['image_info'].attrs['direction']
            self.image_creator = f['image_info'].attrs['image_creator']

            if 'npix_width' in f['image_info'].attrs:
                self.npix = self.npix_width = f['image_info'].attrs['npix_width']
            if 'npix_height' in f['image_info'].attrs:
                self.npix_height = f['image_info'].attrs['npix_height']

            # Recreate orientation object if it is saved
            if direction == 'orientation':
                normal_vector = f['image_info'].attrs['normal_vector']
                perp_vector1 = f['image_info'].attrs['perp_vector1']
                self.orientation = Orientation(normal_vector=normal_vector,
                                               perp_vector1=perp_vector1)
            else:
                self.orientation = None

        if self.direction == 'orientation':
            self._im = ImageCreator(self, self.center, self.widths, self.orientation)
        else:
            self._im = ImageCreator(self, self.center, self.widths, self.direction)

        self.x_c = self.center[0]
        self.y_c = self.center[1]
        self.z_c = self.center[2]
        self.width_x = self.widths[0]
        self.width_y = self.widths[1]
        self.width_z = self.widths[2]

        self.direction = direction

        self.width = self._im.width
        self.height = self._im.height
        self.depth = self._im.depth
        self.centered_extent = self._im.centered_extent
        self.area = self._im.area
        self.volume = self._im.volume

        if hasattr(self, 'npix_width') and hasattr(self, 'npix_height'):
            pass
        else:
            if len(list(self.keys())) > 0:
                arr_shape = self[list(self.keys())[0]].shape
                self.npix = self.npix_width = arr_shape[1]
                self.npix_height = arr_shape[0]
            else:
                import warnings
                warnings.warn('Unable to read pixel dimensions')

        self.area_per_pixel = self.area / (self.npix_width * self.npix_height)
        self.volume_per_pixel = self.volume / (self.npix_width * self.npix_height)

        self.dw = self.width / self.npix_width
        self.dh = self.height / self.npix_height

        # Get derived images
        self.get_derived_images()

    @np.errstate(divide='ignore', invalid='ignore')
    def get_derived_images(self):
        """
        Calculate images automatically for convenience,
        e.g., 0_MagneticFieldSquaredTimesVolume and 0_Volume
        are divided to obtain a 2D-array with 0_MagneticFieldSquared
        """

        # Get derived images from projection-files
        keys = list(self.keys())
        for key in keys:
            if 'Times' in key:
                # Keys of the form 'MagneticFieldSquaredTimesVolume'
                # are split up
                start, end = key.split('Times')
                if end in keys:
                    self[start] = self[key] / self[end]
                elif start[0:2] + end in keys:
                    self[start] = self[key] / self[start[0:2] + end]

        # Calculate density if we have both masses and volumes
        for p in ['', '0_']:
            if (p + 'Masses' in keys) and (p + 'Volume' in keys):
                self[p + 'Density'] = self[p + 'Masses'] / self[p + 'Volume']

    def get_image_coordinates(self):
        extent = self.extent
        npix_width = self.npix_width
        npix_height = self.npix_height
        width = self.width
        height = self.height

        w = extent[0] + (np.arange(npix_width) + 0.5) * width / npix_width
        h = extent[2] + (np.arange(npix_height) + 0.5) * height / npix_height

        if settings.use_units:
            wu = w.unit_quantity
            ww, hh = np.meshgrid(w.value, h.value)
            ww = ww * wu
            hh = hh * wu
        else:
            ww, hh = np.meshgrid(w, h)

        return ww, hh

    def get_centered_image_coordinates(self):

        extent = self.centered_extent
        npix_width = self.npix_width
        npix_height = self.npix_height
        width = self.width
        height = self.height

        w = extent[0] + (np.arange(npix_width) + 0.5) * width / npix_width
        h = extent[2] + (np.arange(npix_height) + 0.5) * height / npix_height

        if settings.use_units:
            wu = w.unit_quantity
            ww, hh = np.meshgrid(w.value, h.value)
            ww = ww * wu
            hh = hh * wu
        else:
            ww, hh = np.meshgrid(w, h)

        return ww, hh


class Histogram2DReader(PaicosReader):
    """
    This is a subclass of the PaicosReader.

    It reads the additional information stored by a Histogram2D instance
    and makes them accessible as attributes, i.e., colorlabel, normalize,
    logscale, hist2d, centers_x, centers_y.
    """

    def __init__(self, basedir='.', snapnum=None, basename='2d_histogram'):
        """
        See documentation for the PaicosReader.

        Returns a dictionary with additional attributes.
        """

        # The PaicosReader class takes care of most of the loading
        super().__init__(basedir, snapnum, basename=basename,
                         load_all=True)

        with h5py.File(self.filename, 'r') as hdf5file:
            if 'colorlabel' in hdf5file['hist2d'].attrs.keys():
                self.colorlabel = hdf5file['hist2d'].attrs['colorlabel']
            self.normalize = hdf5file['hist_info'].attrs['normalize']
            self.logscale = hdf5file['hist_info'].attrs['logscale']

        self.hist2d = self['hist2d']
        self.centers_x = self['centers_x']
        self.centers_y = self['centers_y']
