import numpy as np


def get_variable(snap, variable_str):
    if type(variable_str) is str:
        if variable_str == 'Masses':
            snap.load_data(0, 'Masses')
            variable = snap.P['0_Masses']
        elif variable_str == 'GFM_MetallicityTimesMasses':
            snap.load_data(0, 'GFM_Metallicity')
            variable = snap.P['0_Masses']*snap.P['0_GFM_Metallicity']
        elif variable_str == 'Volumes':
            snap.get_volumes()
            variable = snap.P['0_Volumes']
        elif variable_str == 'EnergyDissipation':
            snap.load_data(0, 'EnergyDissipation')
            variable = snap.P['0_EnergyDissipation']
        elif variable_str == 'MachnumberTimesEnergyDissipation':
            snap.load_data(0, 'Machnumber')
            snap.load_data(0, 'EnergyDissipation')
            variable = snap.P['0_Machnumber']*snap.P['0_EnergyDissipation']
        elif variable_str == 'MagneticFieldSquaredTimesVolume':
            snap.get_volumes()
            snap.load_data(0, 'MagneticField')
            variable = snap.P["0_Volumes"]*np.sum(snap.P['0_MagneticField']**2, axis=1)
        elif variable_str == 'PressureTimesVolume':
            snap.load_data(0, 'InternalEnergy')
            snap.load_data(0, 'Density')
            gamma = 5/3
            # thermal pressure times volume
            # variable = snap.P["0_Volumes"] * snap.P["0_InternalEnergy"] * snap.P["0_Density"] * (gamma - 1.)
            # Same as above but faster
            variable = snap.P["0_Masses"] * snap.P["0_InternalEnergy"] * (gamma - 1.)

        elif variable_str == 'TemperatureTimesMasses':
            snap.get_temperatures()
            variable = snap.P["0_Temperatures"]*snap.P['0_Masses']

        elif variable_str == 'Current':
            snap.load_data(0, 'BfieldGradient')

            def get_index(ii, jj):
                return ii*3 + jj
            gradB = snap.P['0_BfieldGradient'][()]
            J_x = gradB[:, get_index(2, 1)] - gradB[:, get_index(1, 2)]
            J_y = gradB[:, get_index(0, 2)] - gradB[:, get_index(2, 0)]
            J_z = gradB[:, get_index(1, 0)] - gradB[:, get_index(0, 1)]

            J = np.sqrt(J_x**2 + J_y**2 + J_z**2)
            variable = J

        elif variable_str == 'Enstrophy':
            # absolute vorticity squared times one half ("enstrophy")
            snap.load_data(0, 'VelocityGradient')

            def get_index(ii, jj):
                return ii*3 + jj
            gradV = snap.P['0_VelocityGradient'][()]
            vor_x = gradV[:, get_index(2, 1)] - gradV[:, get_index(1, 2)]
            vor_y = gradV[:, get_index(0, 2)] - gradV[:, get_index(2, 0)]
            vor_z = gradV[:, get_index(1, 0)] - gradV[:, get_index(0, 1)]

            enstrophy = 0.5 * (vor_x**2 + vor_y**2 + vor_z**2)
            variable = enstrophy

        elif variable_str == 'EnstrophyTimesMasses':
            # absolute vorticity squared times one half ("enstrophy")
            snap.load_data(0, 'VelocityGradient')

            # Reshaping is slow
            if False:
                n_cells = snap.P['0_VelocityGradient'].shape[0]

                # Reshape to tensor form
                gradV = snap.P['0_VelocityGradient'][()].reshape(n_cells, 3, 3)
                # Get vorticity components
                vor_x = gradV[:, 2, 1] - gradV[:, 1, 2]
                vor_y = gradV[:, 0, 2] - gradV[:, 2, 0]
                vor_z = gradV[:, 1, 0] - gradV[:, 0, 1]
            else:
                def get_index(ii, jj):
                    return ii*3 + jj
                gradV = snap.P['0_VelocityGradient'][()]
                vor_x = gradV[:, get_index(2, 1)] - gradV[:, get_index(1, 2)]
                vor_y = gradV[:, get_index(0, 2)] - gradV[:, get_index(2, 0)]
                vor_z = gradV[:, get_index(1, 0)] - gradV[:, get_index(0, 1)]
            # The vorticity vector
            # vorticity = np.stack([vor_x, vor_y, vor_z], axis=1)

            enstrophy = 0.5 * (vor_x**2 + vor_y**2 + vor_z**2)
            variable = enstrophy*snap.P['0_Masses']
        else:
            raise RuntimeError('unknown function requested', variable_str)

        return variable
