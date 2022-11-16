import h5py
import numpy as np


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
        from astropy import units as u

        with h5py.File(hdf5file, 'r') as f:
            scale_factor = f['Header'].attrs['Time']
            unit_length = f['Parameters'].attrs['UnitLength_in_cm'] * u.cm
            unit_mass = f['Parameters'].attrs['UnitMass_in_g'] * u.g
            unit_velocity = f['Parameters'].attrs['UnitVelocity_in_cm_per_s'] * u.cm/u.s
            unit_time = unit_length / unit_velocity
            unit_energy = unit_mass * unit_velocity**2
            unit_pressure = (unit_mass/unit_length) / unit_time**2
            unit_density = unit_mass/unit_length**3

            HubbleParam = f['Parameters'].attrs['HubbleParam']

        self.arepo_units = {'unit_length': unit_length,
                            'unit_mass': unit_mass,
                            'unit_velocity': unit_velocity,
                            'unit_time': unit_time,
                            'unit_energy': unit_energy,
                            'unit_pressure': unit_pressure,
                            'unit_density': unit_density}
        self.a = scale_factor
        self.h = HubbleParam

    def to_physical(self, name, data):
        """
        Convert arepo data from comoving to physical values.

        name: either a string or a dictionary. If a string, then it should
        data is a numpy array.

        Input names are the arepo hdf5 dataset names.
        """

        if type(name) is dict:
            if ('a_scaling' in name) and ('h_scaling' in name):
                return data * self.a**(name['a_scaling']) * self.h**(name['h_scaling'])
            else:
                print('name:', name)
                raise RuntimeError('name does not contain the required info')

        if name == 'Coordinates':
            return data * self.a / self.h
        elif name == 'Density':
            return data * self.h**2 / self.a**3
        elif name == 'Volume':
            return data * self.a**3 / self.h**3.
        elif name == 'Masses':
            return data/self.h
        elif name == 'EnergyDissipation':
            return data/self.h
        elif name == 'InternalEnergy':
            return data
        elif name == 'MagneticField':
            return data * self.h / self.a**2
        elif name == 'Velocities':
            return data * self.a**0.5
        else:
            err_msg = 'failed to convert from comoving to physical'
            raise RuntimeError(err_msg)

    def give_units(self, name, data):
        """
        Give arepo data units using the units stored in the Parameter group
        in arepo hdf5 outputs. The standard output is CGS using astropy
        quantities. Converting to other units is straightforward using the
        astropy method .to(). For instance, we convert to kpc
        for 'Coordinates'.

        """

        if type(name) is dict:
            if ('length_scaling' in name) and ('mass_scaling' in name):
                aunits = self.arepo_units
                units = aunits['unit_length']**(name['length_scaling']) * \
                    aunits['unit_mass'] **(name['mass_scaling']) * \
                    aunits['unit_velocity']**(name['velocity_scaling'])
                return data * units
            else:
                print('name:', name)
                raise RuntimeError('name does not contain the required info')
        if name == 'Coordinates':
            return data * self.arepo_units['unit_length'].to('kpc')
        elif name == 'Density':
            return data * self.arepo_units['unit_density']
        elif name == 'Volume':
            return data / self.arepo_units['unit_length'].to('kpc')**3
        elif name == 'Masses':
            return data * self.arepo_units['unit_mass'].to('Msun')
        # elif name == 'EnergyDissipation':
        #     return data * self.arepo_units['unit_energy'] # Check!
        elif name == 'InternalEnergy':
            return data * self.arepo_units['unit_energy']/self.arepo_units['unit_mass']
        elif name == 'MagneticField':
            return data * np.sqrt(self.y['unit_pressure'])
        elif name == 'Velocities':
            return data * self.arepo_units['unit_velocity']
        else:
            err_msg = 'failed to give units using astropy'
            raise RuntimeError(err_msg)

    def to_physical_and_give_units(self, name, data):
        """
        This is simply a convenience method which calls two other methods.
        """
        data = self.to_physical(name, data)
        data = self.give_units(name, data)
        return data


if __name__ == '__main__':
    from paicos import get_project_root_dir

    path = get_project_root_dir()

    converter = ArepoConverter(path + '/data/slice_x.hdf5')

    rho = np.array([2, 4])
    rho = converter.to_physical_and_give_units('Density', rho)

    Mstars = 1
    Mstars = converter.to_physical_and_give_units('Masses', Mstars)
