"""
Defines the PaicosReader, Histogram2DReader and ImageReader which
can be used to load derived variables.
"""
import os
import h5py
import numpy as np
from astropy import units as u
from astropy.cosmology import LambdaCDM
from .. import util
from .. import units as pu
from .. import settings


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

        Parameters:

        basedir (str): path of the directory containing the hdf5 files
                       (e.g. the 'output' folder)

        snapnum (int): e.g. snapshot number

        basename (str): name of the file takes the form 'basename_{:03d}.hdf5'
                        or 'basename_{:03d}.{}.hdf5'. Default is 'snap'.

        basesubdir (str): name of the subfolder. Default is "snapdir".

        load_all (bool): whether to simply load all data, default is True

        to_physical (bool): whether to convert from comoving to physical
                            variables upon load, not yet implemented!

        verbose (bool): whether to print information, default is False
        """

        self.basedir = basedir
        self.snapnum = snapnum
        self.basename = basename

        assert to_physical is False, 'to_physical not yet implemented!'

        self.to_physical = to_physical
        self.basesubdir = basesubdir
        self.verbose = verbose

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
        self.enable_units()
        self.add_user_units()

        # Find the adiabatic index
        self.gamma = self.get_adiabiatic_index()

        # Load all data sets
        if load_all:
            for key in keys:
                self.load_data(key)

    def get_units_and_other_parameters(self):
        """
        Define arepo units, scale factor (a) and h (HubbleParam).

        The input required is a hdf5 file with the Parameters and Header
        groups found in arepo snapshots.
        """

        self._Time = self.Header['Time']
        self._Redshift = self.Header['Redshift']
        self._HubbleParam = self.Parameters['HubbleParam']

        unit_length = self.Parameters['UnitLength_in_cm'] * u.cm
        unit_mass = self.Parameters['UnitMass_in_g'] * u.g
        unit_velocity = self.Parameters['UnitVelocity_in_cm_per_s'] * u.cm / u.s
        unit_time = unit_length / unit_velocity
        unit_energy = unit_mass * unit_velocity**2
        unit_pressure = (unit_mass / unit_length) / unit_time**2
        unit_density = unit_mass / unit_length**3
        Omega0 = self.Parameters['Omega0']
        OmegaBaryon = self.Parameters['OmegaBaryon']
        OmegaLambda = self.Parameters['OmegaLambda']

        ComovingIntegrationOn = self.Parameters['ComovingIntegrationOn']

        self.ComovingIntegrationOn = bool(ComovingIntegrationOn)
        self.comoving_sim = self.ComovingIntegrationOn

        if self.ComovingIntegrationOn:
            # Set up LambdaCDM cosmology to calculate times, etc
            self.cosmo = LambdaCDM(H0=100 * self.h, Om0=Omega0,
                                   Ob0=OmegaBaryon, Ode0=OmegaLambda)
            # Current age of the universe and look back time
            self._age = self.get_age(self.z)
            self._lookback_time = self.get_lookback_time(self.z)

        _ns = globals()
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

        self.arepo_units_in_cgs = {'unit_length': unit_length,
                                   'unit_mass': unit_mass,
                                   'unit_velocity': unit_velocity,
                                   'unit_time': unit_time,
                                   'unit_energy': unit_energy,
                                   'unit_pressure': unit_pressure,
                                   'unit_density': unit_density}

        self.arepo_units = {'unit_length': arepo_length,
                            'unit_mass': arepo_mass,
                            'unit_velocity': arepo_velocity,
                            'unit_time': arepo_time,
                            'unit_energy': arepo_energy,
                            'unit_pressure': arepo_pressure,
                            'unit_density': arepo_density}

    def enable_units(self):
        """
        Enables arepo units globally
        """
        for unit_name, unit in self.arepo_units.items():
            u.add_enabled_units(unit)
            phys_type = unit_name.split('_')[1]
            u.def_physical_type(unit, phys_type)

        self.length = self.get_paicos_quantity(1, 'Coordinates')
        self.mass = self.get_paicos_quantity(1, 'Masses')
        self.velocity = self.get_paicos_quantity(1, 'Velocities')

    def add_user_units(self):
        """
        Add all user supplied units
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
        return self._Time * self.arepo_units['unit_time']

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
                time = pu.PaicosTimeSeries(time, a=a, h=self.h,
                                           comoving_sim=self.comoving_sim)
            else:
                time = pu.PaicosQuantity(time, a=a, h=self.h,
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
        Find unit for a given quantity
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
            msg = '\n\nUnit for {}, {} not implemented!'
            msg += '\nPlease add it to unit_specifications'

            if settings.strict_units:
                raise RuntimeError(msg.format(field, name))
            return False

        return self._sanitize_unit(unit)

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

    def load_data(self, name, group=None):
        """
        Load data from a generic Paicos hdf5 file (written by a PaicosWriter
        instance).

        The method requires that the data sets have a 'unit' attribute
        and hence does work for Arepo hdf5 files.
        For this reason, the method is overloaded in the Snapshot and Catalog
        classes.
        """

        with h5py.File(self.filename, 'r') as f:
            if isinstance(f[name], h5py.Dataset):
                self[name] = util.load_dataset(f, name, group=group)
            elif isinstance(f[name], h5py.Group):
                for data_name in f[name].keys():
                    data = util.load_dataset(f, data_name, group=name)
                    if name not in self:
                        self[name] = {}
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

    def __init__(self, basedir, snapnum, basename="projection", load_all=True):
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

        self.x_c = self.center[0]
        self.y_c = self.center[1]
        self.z_c = self.center[2]
        self.width_x = self.widths[0]
        self.width_y = self.widths[1]
        self.width_z = self.widths[2]

        self.direction = direction

        if direction == 'x':

            self.width = self.width_y
            self.height = self.width_z
            self.depth = self.width_x

        elif direction == 'y':

            self.width = self.width_x
            self.height = self.width_z
            self.depth = self.width_y

        elif direction == 'z':

            self.width = self.width_x
            self.height = self.width_y
            self.depth = self.width_z

        self.centered_extent = self.extent.copy
        self.centered_extent[0] = -self.width / 2
        self.centered_extent[1] = +self.width / 2
        self.centered_extent[2] = -self.height / 2
        self.centered_extent[3] = +self.height / 2

        area = (self.extent[1] - self.extent[0]) * (self.extent[3] - self.extent[2])
        self.area = area

        if len(list(self.keys())) > 0:
            arr_shape = self[list(self.keys())[0]].shape
            self.npix = self.npix_width = arr_shape[0]
            self.npix_height = arr_shape[1]
        self.area_per_pixel = self.area / (self.npix_width * self.npix_height)
        self.volume = self.width_x * self.width_y * self.width_z
        self.volume_per_pixel = self.volume / (self.npix_width * self.npix_height)

        self.dw = self.width / self.npix_width
        self.dh = self.height / self.npix_height

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

    def __init__(self, basedir, snapnum, basename='2d_histogram'):
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
