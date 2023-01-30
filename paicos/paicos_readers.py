from . import util
import h5py
import numpy as np
import os
from astropy import units as u
from . import units as pu
from astropy.cosmology import LambdaCDM
from . import settings


class PaicosReader(dict):

    def __init__(self, basedir='.', snapnum=None, basename="snap",
                 load_all=True, to_physical=False,
                 basesubdir='snapdir', verbose=False):

        self.basedir = basedir
        self.snapnum = snapnum
        self.basename = basename

        self.to_physical = to_physical
        self.basesubdir = basesubdir
        self.verbose = verbose

        if basedir[-1] != '/':
            basedir += '/'

        if snapnum is None:
            self.filename = basedir + basename + '.hdf5'
            msg = 'File: {} not found found'.format(self.filename)
            assert os.path.exists(self.filename), msg
        else:
            single_file = basename + "_{:03d}.hdf5"
            multi_file = basesubdir + '_{:03d}/' + basename + '_{:03d}.{}.hdf5'
            multi_wo_dir = basename + '_{:03d}.{}.hdf5'
            #

            single_file = basedir + single_file.format(snapnum)
            multi_file = basedir + multi_file.format(snapnum, snapnum, '{}')
            multi_wo_dir = basedir + multi_wo_dir.format(snapnum, '{}')

            if os.path.exists(single_file):
                self.multi_file = False
                self.filename = single_file
            elif os.path.exists(multi_file.format(0)):
                self.multi_file = True
                self.first_file_name = self.filename = multi_file.format(0)
                self.no_subdir = False
            elif os.path.exists(multi_wo_dir.format(0)):
                self.multi_file = True
                self.first_file_name = self.filename = multi_wo_dir.format(0)
                self.no_subdir = True
            else:
                err_msg = "File: {} not found found"
                raise FileNotFoundError(err_msg.format(self.file_name))

        with h5py.File(self.filename, 'r') as f:
            self.Header = dict(f['Header'].attrs)
            self.Config = dict(f['Config'].attrs)
            self.Parameters = dict(f['Parameters'].attrs)
            keys = list(f.keys())

        # Enable units
        self.get_units_and_other_parameters()
        self.enable_units()

        # Find the adiabatic index
        if 'GAMMA' in self.Config:
            self.gamma = self.Config['GAMMA']
        elif 'ISOTHERMAL' in self.Config:
            self.gamma = 1
        else:
            self.gamma = 5/3

        # Load all data sets
        if load_all:
            for key in keys:
                self.load_data(key)

    def get_units_and_other_parameters(self):
        """
        Initialize arepo units, scale factor (a) and h (HubbleParam).

        The input required is a hdf5 file with the Parameters and Header
        groups found in arepo snapshots.
        """

        self._Time = self.Header['Time']
        self._Redshift = self.Header['Redshift']
        self._HubbleParam = self.Parameters['HubbleParam']

        unit_length = self.Parameters['UnitLength_in_cm'] * u.cm
        unit_mass = self.Parameters['UnitMass_in_g'] * u.g
        unit_velocity = self.Parameters['UnitVelocity_in_cm_per_s'] * u.cm/u.s
        unit_time = unit_length / unit_velocity
        unit_energy = unit_mass * unit_velocity**2
        unit_pressure = (unit_mass/unit_length) / unit_time**2
        unit_density = unit_mass/unit_length**3
        Omega0 = self.Parameters['Omega0']
        OmegaBaryon = self.Parameters['OmegaBaryon']
        OmegaLambda = self.Parameters['OmegaLambda']

        ComovingIntegrationOn = self.Parameters['ComovingIntegrationOn']

        self.ComovingIntegrationOn = bool(ComovingIntegrationOn)
        self.comoving_sim = self.ComovingIntegrationOn

        if self.ComovingIntegrationOn:
            # Set up LambdaCDM cosmology to calculate times, etc
            self.cosmo = LambdaCDM(H0=100*self.h, Om0=Omega0,
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
        # Enable arepo units globally
        for key in self.arepo_units:
            u.add_enabled_units(self.arepo_units[key])
            phys_type = key.split('_')[1]
            u.def_physical_type(self.arepo_units[key], phys_type)

        self.length = self.get_paicos_quantity(1, 'Coordinates')
        self.mass = self.get_paicos_quantity(1, 'Masses')
        self.velocity = self.get_paicos_quantity(1, 'Velocities')

    @property
    def a(self):
        """
        The scale factor.
        """
        if self.comoving_sim:
            return self._Time
        else:
            raise RuntimeError('Non-comoving object has no scale factor')
            # return 1.

    @property
    def h(self):
        """
        The reduced Hubble parameter
        """
        if self._HubbleParam == 0:
            return 1.0
        else:
            return self._HubbleParam

    @property
    def z(self):
        """
        The redshift.
        """
        if self.comoving_sim:
            return self._Redshift
        else:
            raise RuntimeError('Non-comoving object has no redshift')
            # return 0.

    @property
    def lookback_time(self):
        if self.comoving_sim:
            return self._lookback_time(self.z)
        else:
            raise RuntimeError('Non-comoving object has no lookback_time')

    @property
    def age(self):
        if self.comoving_sim:
            return self._age
        else:
            raise RuntimeError('Non-comoving object has no lookback_time')

    @property
    def time(self):
        if self.comoving_sim:
            return self.age
        else:
            return self._Time * self.arepo_units['unit_time']

    def get_lookback_time(self, z):
        lookback_time = self.cosmo.lookback_time(z)

        return self.__convert_to_paicos(lookback_time, z)

    def get_age(self, z):
        age = self.cosmo.lookback_time(1e100) - self.cosmo.lookback_time(z)

        return self.__convert_to_paicos(age, z)

    def __convert_to_paicos(self, time, z):
        if settings.use_units:
            a = 1.0/(z + 1.)
            if isinstance(a, np.ndarray):
                time = pu.PaicosTimeSeries(time, a=a, h=self.h,
                                           comoving_sim=self.comoving_sim)
            else:
                time = pu.PaicosQuantity(time, a=a, h=self.h,
                                         comoving_sim=self.comoving_sim)
        return time

    def get_paicos_quantity(self, data, name):

        if hasattr(data, 'unit'):
            msg = 'Data already had units! {}'.format(name)
            raise RuntimeError(msg)

        unit = self.find_unit(name)

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        return pu.PaicosQuantity(data, unit, a=self._Time, h=self.h,
                                 comoving_sim=self.comoving_sim,
                                 dtype=data.dtype)

    def find_unit(self, name, arepo_code_units=True):
        """
        Here we find the units including the scaling with a and h
        of a quantity.

        The input 'name' can be either a data attribute
        from arepo (which currently does not exist for all variables)
        or it can be a string corresponding to one of the data types,
        i.e. 'Velocities' or 'Coordinates'

        For this latter, hardcoded, option, I have implemented a few of the
        gas variables.
        """
        import astropy.units as u
        if arepo_code_units:
            aunits = self.arepo_units
        else:
            aunits = self.arepo_units_in_cgs

        # Turn off a and h if we are not comoving or if h = 1
        if self.ComovingIntegrationOn:
            a = pu.small_a
        else:
            a = u.Unit('')

        if self.h == 1:
            h = u.Unit('')
        else:
            h = pu.small_h

        if arepo_code_units:
            def find(name):
                return self.find_unit(name, True)
        else:
            def find(name):
                return self.find_unit(name, False)

        if isinstance(name, dict):
            # Create units for the quantity
            if 'unit' in name:
                units = 1*u.Unit(name['unit'])
            else:
                # Arepo data attributes and the units from the Parameter
                # group in the hdf5 file are here combined
                # Create comoving dictionary
                comoving_dic = {}
                for key in ['a_scaling', 'h_scaling']:
                    comoving_dic.update({key: name[key]})

                comoving_dic.update({'small_h': self.h,
                                     'scale_factor': self.a})
                units = aunits['unit_length']**(name['length_scaling']) * \
                    aunits['unit_mass']**(name['mass_scaling']) * \
                    aunits['unit_velocity']**(name['velocity_scaling']) * \
                    a**comoving_dic['a_scaling'] * \
                    h**comoving_dic['h_scaling']

        elif isinstance(name, str):

            unitless_vars = ['ElectronAbundance', 'MachNumber',
                             'GFM_Metallicity', 'GFM_Metals', 'ParticleIDs']
            if name == 'Coordinates':
                units = aunits['unit_length']*a/h
            elif name == 'Density':
                units = find('Masses')/find('Volume')
            elif name == 'Volume':
                units = find('Coordinates')**3
            elif name in unitless_vars:
                units = ''
            elif name == 'Masses':
                units = aunits['unit_mass']/h
            elif name == 'EnergyDissipation':
                units = aunits['unit_energy']/h
                raise RuntimeError('Needs checking!')
            elif name == 'InternalEnergy':
                units = aunits['unit_energy']/aunits['unit_mass']
            elif name == 'MagneticField':
                units = aunits['unit_pressure']**(1/2)*a**(-2)*h
            elif name == 'BfieldGradient':
                units = find('MagneticField')/find('Coordinates')
            elif name == 'MagneticFieldDivergence':
                units = find('BfieldGradient')
            elif name == 'Velocities':
                units = aunits['unit_velocity']*a**(1/2)
            elif name == 'Velocities':
                units = aunits['unit_velocity']*a**(1/2)
            elif name == 'VelocityGradient':
                units = find('Velocities')/find('Coordinates')
            elif name == 'Enstrophy':
                units = (find('VelocityGradient'))**2
            elif name == 'Temperature':
                units = u.K
            elif name == 'Pressure':
                units = aunits['unit_pressure'] * h**2 / a**3
            else:
                err_msg = 'invalid option name={}, cannot find units'
                raise RuntimeError(err_msg.format(name))

        return units

    def load_data(self, name, group=None):

        with h5py.File(self.filename, 'r') as f:
            if isinstance(f[name], h5py._hl.dataset.Dataset):
                self[name] = util.load_dataset(f, name, group=group)
            elif isinstance(f[name], h5py._hl.group.Group):
                for data_name in f[name].keys():
                    data = util.load_dataset(f, data_name, group=name)
                    if name not in self:
                        self[name] = {}
                    self[name][data_name] = data


class ImageReader(PaicosReader):

    def __init__(self, basedir, snapnum, basename="projection", load_all=True):

        # The PaicosReader class takes care of most of the loading
        super().__init__(basedir, snapnum, basename=basename,
                         load_all=load_all)

        # Load info specific to images
        with h5py.File(self.filename, 'r') as f:
            self.extent = util.load_dataset(f, 'extent', group='image_info')
            self.widths = util.load_dataset(f, 'widths', group='image_info')
            self.center = util.load_dataset(f, 'center', group='image_info')
            self.direction = f['image_info'].attrs['direction']
            self.image_creator = f['image_info'].attrs['image_creator']

        # Get derived images from projection-files
        keys = list(self.keys())
        for key in keys:
            if 'Times' in key:
                # Keys of the form 'MagneticFieldSquaredTimesVolume'
                # are split up
                start, end = key.split('Times')
                if (end in keys):
                    self[start] = self[key]/self[end]
                elif (start[0:2] + end in keys):
                    self[start] = self[key]/self[start[0:2] + end]

        # Calculate density if we have both masses and volumes
        for p in ['', '0_']:
            if (p + 'Masses' in keys) and (p + 'Volume' in keys):
                self[p + 'Density'] = self[p+'Masses']/self[p+'Volume']


class Histogram2DReader(PaicosReader):

    def __init__(self, basedir, snapnum, basename='2d_histogram'):

        # The PaicosReader class takes care of most of the loading
        super().__init__(basedir, snapnum, basename=basename,
                         load_all=True)

        with h5py.File(self.filename, 'r') as hdf5file:
            if 'colorlabel' in hdf5file['hist2d'].attrs.keys():
                self.colorlabel = hdf5file['hist2d'].attrs['colorlabel']
            self.normalize = hdf5file['hist_info'].attrs['normalize']
            self.logscale = hdf5file['hist_info'].attrs['normalize']

        self.hist2d = self['hist2d']
        self.centers_x = self['centers_x']
        self.centers_y = self['centers_y']
