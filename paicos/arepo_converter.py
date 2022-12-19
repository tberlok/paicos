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

        _ns = globals()
        arepo_mass = u.def_unit(
            ["arepo_mass", "a_mass"],
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

        # Enable arepo units globally
        for key in self.arepo_units:
            u.add_enabled_units(self.arepo_units[key])
            phys_type = key.split('_')[1]
            u.def_physical_type(self.arepo_units[key], phys_type)

        self.a = self.scale_factor = scale_factor
        self.z = self.redshift = redshift
        self.h = HubbleParam

        self.length = self.get_paicos_quantity(1, 'Coordinates')
        self.mass = self.get_paicos_quantity(1, 'Masses')
        self.velocity = self.get_paicos_quantity(1, 'Velocities')

        # Set up LambdaCDM cosmology to calculate times, etc
        self.cosmo = cosmo = LambdaCDM(H0=100*HubbleParam, Om0=Omega0,
                                       Ob0=OmegaBaryon, Ode0=OmegaLambda)
        # Current age of the universe and look back time
        self.age = cosmo.lookback_time(1e100) - cosmo.lookback_time(self.z)
        self.lookback_time = cosmo.lookback_time(self.z)

    def get_paicos_quantity(self, data, name, arepo_code_units=True):

        if hasattr(data, 'unit'):
            print('Data already has units, returning!')
            return data

        unit = self.find_unit(name, arepo_code_units)

        data = np.array(data)

        return pu.PaicosQuantity(data, unit, a=self.a, h=self.h)

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
        if arepo_code_units:
            aunits = self.arepo_units
        else:
            aunits = self.arepo_units_in_cgs
        a = pu.small_a
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
            if name == 'Coordinates':
                units = aunits['unit_length']*a/h
            elif name == 'Density':
                units = find('Masses')/find('Volumes')
            elif name == 'Volumes':
                units = find('Coordinates')**3
            elif name == 'Masses':
                units = aunits['unit_mass']/h
            elif name == 'EnergyDissipation':
                units = aunits['unit_energy']/h
            elif name == 'InternalEnergy':
                units = aunits['unit_energy']/aunits['unit_mass']
            elif name == 'MagneticField':
                units = aunits['unit_pressure']**(1/2)*a**(-2)*h
            elif name == 'BfieldGradient':
                units = find('MagneticField')/find('Coordinates')
            elif name == 'MagneticField':
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
                comoving_dic = {}
                units = u.K
            else:
                err_msg = 'invalid option name={}, cannot find units'
                raise RuntimeError(err_msg.format(name))

        return units

    def give_units(self, name, data):
        """
        For backward compatibility, this method returns
        an astropy object. Might be removed in the future.
        """
        if hasattr(data, 'unit'):
            raise RuntimeError('error in input')
        quantity = self.get_paicos_quantity(data, name)
        return quantity.value*quantity.unit

    def to_physical(self, name, data):
        """
        For backward compatibility, this method returns a
        numerical value. Might be removed in the future.
        """
        if hasattr(data, 'unit'):
            raise RuntimeError('error in input')
        quantity = self.get_paicos_quantity(data, name)
        quantity = quantity.to_physical
        return quantity.value

    def to_physical_and_give_units(self, name, data):
        """
        For backward compatibility, this method returns
        an astropy object. Might be removed in the future.
        """
        if hasattr(data, 'unit'):
            raise RuntimeError('error in input')
        quantity = self.get_paicos_quantity(data, name)
        quantity = quantity.to_physical
        # Return an astropy quantity
        return quantity.value*quantity.unit


if __name__ == '__main__':
    from paicos import root_dir

    converter = ArepoConverter(root_dir + '/data/slice_x.hdf5')

    rho = np.array([2, 4])
    rho = converter.get_paicos_quantity(rho, 'Density')

    Mstars = 1
    Mstars = converter.get_paicos_quantity(Mstars, 'Masses')

    B = converter.get_paicos_quantity([1], 'MagneticField')

    T = converter.get_paicos_quantity([1], 'Temperature')

    B = converter.get_paicos_quantity([1], 'MagneticField')

    v = converter.get_paicos_quantity([2], 'Velocities')

    mv_stars = Mstars*v

    print(T.to('keV'))
    print(B.to('uG'))
