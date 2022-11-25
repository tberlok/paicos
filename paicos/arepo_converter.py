import h5py
import numpy as np
from astropy import units as u
from paicos import units as pu
from astropy.cosmology import LambdaCDM


class ArepoConverter:
    """
    This class implements functionality for
    1) converting from the comoving units used in Arepo.
    2) converting from internal code units to physical units (CGS) using
       astropy.

    Methods are:

    to_physical(name, data)
    give_units(name, data)
    to_physical_and_give_units(name, data)

    Example code:
     converter = ArepoConverter('thin_projection_z_247.hdf5')

     rho = np.array([2, 4])
     rho = converter.to_physical_and_give_units('Density', rho)

     Mstars = 1
     Mstars = converter.to_physical_and_give_units('Masses', Mstars)

    """
    def __init__(self, hdf5file):
        """
        Initialize arepo units, scale factor (a) and h (HubbleParam).

        The input required is a hdf5 file with the Parameters and Header
        groups found in arepo snapshots.
        """

        # Allows conversion between K and eV
        u.add_enabled_equivalencies(u.temperature_energy())
        # Allows conversion to Gauss (potential issues?)
        # https://github.com/astropy/astropy/issues/7396
        gauss_B = (u.g/u.cm)**(0.5)/u.s
        equiv_B = [(u.G, gauss_B, lambda x: x, lambda x: x)]
        u.add_enabled_equivalencies(equiv_B)

        with h5py.File(hdf5file, 'r') as f:
            scale_factor = f['Header'].attrs['Time']
            redshift = f['Header'].attrs['Redshift']
            unit_length = f['Parameters'].attrs['UnitLength_in_cm'] * u.cm
            unit_mass = f['Parameters'].attrs['UnitMass_in_g'] * u.g
            unit_velocity = f['Parameters'].attrs['UnitVelocity_in_cm_per_s'] * u.cm/u.s
            unit_time = unit_length / unit_velocity
            unit_energy = unit_mass * unit_velocity**2
            unit_pressure = (unit_mass/unit_length) / unit_time**2
            unit_density = unit_mass/unit_length**3
            Omega0 = f['Header'].attrs['Omega0']
            OmegaBaryon = f['Header'].attrs['OmegaBaryon']
            OmegaLambda = f['Header'].attrs['OmegaLambda']

            HubbleParam = f['Parameters'].attrs['HubbleParam']

        self.arepo_units = {'unit_length': unit_length,
                            'unit_mass': unit_mass,
                            'unit_velocity': unit_velocity,
                            'unit_time': unit_time,
                            'unit_energy': unit_energy,
                            'unit_pressure': unit_pressure,
                            'unit_density': unit_density}
        self.a = self.scale_factor = scale_factor
        self.z = self.redshift = redshift
        self.h = HubbleParam

        # Set up LambdaCDM cosmology to calculate times, etc
        self.cosmo = cosmo = LambdaCDM(H0=100*HubbleParam, Om0=Omega0,
                                       Ob0=OmegaBaryon, Ode0=OmegaLambda)
        # Current age of the universe and look back time
        self.age = cosmo.lookback_time(1e100) - cosmo.lookback_time(self.z)
        self.lookback_time = cosmo.lookback_time(self.z)

    def to_physical(self, name, data):
        """
        Convert arepo data from comoving to physical values.

        name: either a string or a dictionary. If a string, then it should
        data is a numpy array.

        Input names are the arepo hdf5 dataset names.
        """

        comoving_dic, units = self.get_comoving_dic_and_units(name)

        if type(data) is list:
            data = np.array(data)

        factor = self.a**(comoving_dic['a_scaling']) * \
            self.h**(comoving_dic['h_scaling'])

        return data * factor

    def give_units(self, name, data):
        """
        Give arepo data units using the units stored in the Parameter group
        in arepo hdf5 outputs. The standard output is CGS using astropy
        quantities. Converting to other units is straightforward using the
        astropy method .to(). For instance, we convert to kpc
        for 'Coordinates'.

        """
        comoving_dic, units = self.get_comoving_dic_and_units(name)

        if type(data) is list:
            data = np.array(data)
        return data*units

    def to_physical_and_give_units(self, name, data):
        """
        This is simply a convenience method which calls two other methods.
        """
        data = self.to_physical(name, data)
        data = self.give_units(name, data)
        return data

    def get_comoving_quantity(self, name, data):
        from paicos import ComovingQuantity

        comoving_dic, units = self.get_comoving_dic_and_units(name)

        data = np.array(data)

        return ComovingQuantity(data, comoving_dic=comoving_dic)*units

    def get_paicos_quantity(self, name, data):

        comoving_dic, u_units = self.get_comoving_dic_and_units(name)

        data = np.array(data)

        pu_units = pu.small_a**comoving_dic['a_scaling'] * \
            pu.small_h**comoving_dic['h_scaling']

        return pu.PaicosQuantity(data, u_units*pu_units, a=self.a, h=self.h)

    def get_comoving_dic_and_units(self, name):
        """
        Here we find the units and the scaling with a and h
        of a quantity.

        The input 'name' can be either a data attribute
        from arepo (which currently does not exist for all variables)
        or it can be a string corresponding to one of the data types,
        i.e. 'Velocities' or 'Coordinates'

        For this latter, hardcoded, option, I have implemented a few of the
        gas variables.
        """

        if isinstance(name, dict):
            # Create comoving dictionary
            comoving_dic = {}
            for key in ['a_scaling', 'h_scaling']:
                comoving_dic.update({key: name[key]})

            comoving_dic.update({'small_h': self.h,
                                 'scale_factor': self.a})
            # Create units for the quantity
            if 'units' in name:
                units = 1*u.Unit(name['units'])
            else:
                # Arepo data attributes and the units from the Parameter
                # group in the hdf5 file are here combined
                aunits = self.arepo_units
                units = aunits['unit_length']**(name['length_scaling']) * \
                    aunits['unit_mass']**(name['mass_scaling']) * \
                    aunits['unit_velocity']**(name['velocity_scaling'])

        elif isinstance(name, str):
            if name == 'Coordinates':
                comoving_dic = {'a_scaling': 1, 'h_scaling': -1}
                units = self.arepo_units['unit_length'].to('kpc')
            elif name == 'Density':
                comoving_dic = {'a_scaling': -3, 'h_scaling': 2}
                units = self.arepo_units['unit_density']
            elif name == 'Volume':
                comoving_dic = {'a_scaling': 3, 'h_scaling': -3}
                units = 1/self.arepo_units['unit_length'].to('kpc')**3
            elif name == 'Masses':
                comoving_dic = {'h_scaling': -1}
                units = self.arepo_units['unit_mass'].to('Msun')
            elif name == 'EnergyDissipation':
                comoving_dic = {'h_scaling': -1}
                units = self.arepo_units['unit_energy']
            elif name == 'InternalEnergy':
                comoving_dic = {}
                units = self.arepo_units['unit_energy']/self.arepo_units['unit_mass']
            elif name == 'MagneticField':
                comoving_dic = {'a_scaling': -2, 'h_scaling': 1}
                units = np.sqrt(self.arepo_units['unit_pressure'])
            elif name == 'Velocities':
                comoving_dic = {'a_scaling': 0.5}
                units = self.arepo_units['unit_velocity']
            elif name == 'Temperature':
                comoving_dic = {}
                units = u.K
            else:
                err_msg = 'invalid option name={}, cannot find units'
                raise RuntimeError(err_msg.format(name))
            for key in ['a_scaling', 'h_scaling']:
                if key not in comoving_dic:
                    comoving_dic.update({key: 0})

            # Finally add
            comoving_dic.update({'small_h': self.h,
                                 'scale_factor': self.a})
        return comoving_dic, units


if __name__ == '__main__':
    from paicos import root_dir

    converter = ArepoConverter(root_dir + '/data/slice_x.hdf5')

    rho = np.array([2, 4])
    rho = converter.to_physical_and_give_units('Density', rho)

    Mstars = 1
    Mstars = converter.to_physical_and_give_units('Masses', Mstars)

    B = converter.to_physical_and_give_units('MagneticField', [1])

    T = converter.give_units('Temperature', 1)

    B = converter.to_physical_and_give_units('MagneticField', [1])

    print(T.to('keV'))
    print(B.to('uG'))

    B = converter.get_comoving_quantity('MagneticField', [1])
    B_alt = converter.get_paicos_quantity('MagneticField', [1])
