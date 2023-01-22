import numpy as np


def get_variable_function_gas(variable_str, info=False):

    def GFM_MetallicityTimesMasses(snap):
        return snap['0_Masses']*snap['0_GFM_Metallicity']

    def Volumes(snap):
        return snap["0_Masses"] / snap["0_Density"]

    def EnergyDissipation(snap):
        return snap['0_EnergyDissipation']

    def MachnumberTimesEnergyDissipation(snap):
        variable = snap['0_Machnumber']*snap['0_EnergyDissipation']
        return variable

    def MagneticFieldSquared(snap):
        return np.sum(snap['0_MagneticField']**2, axis=1)

    def MagneticFieldSquaredTimesVolumes(snap):
        variable = snap["0_Volumes"]*np.sum(snap['0_MagneticField']**2, axis=1)
        return variable

    def PressureTimesVolumes(snap):
        gamma = 5/3
        # TODO: Get rid of hardcoded gamma
        # thermal pressure times volume
        # variable = snap["0_Volumes"] * snap["0_InternalEnergy"] * snap["0_Density"] * (gamma - 1.)
        # Same as above but faster
        variable = snap["0_Masses"] * snap["0_InternalEnergy"] * (gamma - 1.)
        return variable

    def Temperatures2(snap):
        from astropy import constants as c
        mhydrogen = c.m_e + c.m_p

        if 'GAMMA' in snap.Config:
            gamma = snap.Config['GAMMA']
        elif 'ISOTHERMAL' in snap.Config:
            msg = 'temperature is constant when ISOTHERMAL in Config'
            raise RuntimeError(msg)
        else:
            gamma = 5/3

        gm1 = gamma - 1

        mmean = snap['0_MeanMolecularWeight']

        # temperature in Kelvin
        from . import settings
        if settings.use_units:
            variable = (gm1 * snap["0_InternalEnergy"] *
                        mmean * mhydrogen).to('K')
        else:
            u_v = snap.converter.arepo_units['unit_velocity']
            variable = (gm1 * snap["0_InternalEnergy"] *
                        u_v**2 * mmean * mhydrogen
                        ).to('K').value
        return variable

    def Temperatures(snap):
        from astropy import constants as c
        mhydrogen = c.m_e + c.m_p

        fhydrogen = 0.76

        if "ElectronAbundance" in snap.info(0, False):
            mmean = 4.0 / (1.0 + 3.0*fhydrogen + 4.0 *
                           fhydrogen*snap["0_ElectronAbundance"])
        else:
            mmean_ionized = (1.0+(1.0-fhydrogen)/fhydrogen) / \
                (2.0+3.0*(1.0-fhydrogen)/(4.0*fhydrogen))
            mmean = mmean_ionized

        if 'GAMMA' in snap.Config:
            gamma = snap.Config['GAMMA']
        elif 'ISOTHERMAL' in snap.Config:
            msg = 'temperature is constant when ISOTHERMAL in Config'
            raise RuntimeError(msg)
        else:
            gamma = 5/3

        gm1 = gamma - 1

        # temperature in Kelvin
        from . import settings
        if settings.use_units:
            variable = (gm1 * snap["0_InternalEnergy"] *
                        mmean * mhydrogen).to('K')
        else:
            u_v = snap.converter.arepo_units['unit_velocity']
            variable = (gm1 * snap["0_InternalEnergy"] *
                        u_v**2 * mmean * mhydrogen
                        ).to('K').value
        return variable

    def TemperaturesTimesMasses(snap):
        return snap["0_Temperatures"]*snap['0_Masses']

    def Current(snap):
        snap.load_data(0, 'BfieldGradient')

        def get_index(ii, jj):
            return ii*3 + jj
        gradB = snap['0_BfieldGradient'][()]
        J_x = gradB[:, get_index(2, 1)] - gradB[:, get_index(1, 2)]
        J_y = gradB[:, get_index(0, 2)] - gradB[:, get_index(2, 0)]
        J_z = gradB[:, get_index(1, 0)] - gradB[:, get_index(0, 1)]

        J = np.sqrt(J_x**2 + J_y**2 + J_z**2)
        return J

    def Enstrophy(snap):
        # absolute vorticity squared times one half ("enstrophy")

        def get_index(ii, jj):
            return ii*3 + jj
        gradV = snap['0_VelocityGradient'][()]
        vor_x = gradV[:, get_index(2, 1)] - gradV[:, get_index(1, 2)]
        vor_y = gradV[:, get_index(0, 2)] - gradV[:, get_index(2, 0)]
        vor_z = gradV[:, get_index(1, 0)] - gradV[:, get_index(0, 1)]

        enstrophy = 0.5 * (vor_x**2 + vor_y**2 + vor_z**2)
        return enstrophy

    def EnstrophyTimesMasses(snap):
        # absolute vorticity squared times one half ("enstrophy")

        # Reshaping is slow
        if False:
            n_cells = snap['0_VelocityGradient'].shape[0]

            # Reshape to tensor form
            gradV = snap['0_VelocityGradient'].reshape(n_cells, 3, 3)
            # Get vorticity components
            vor_x = gradV[:, 2, 1] - gradV[:, 1, 2]
            vor_y = gradV[:, 0, 2] - gradV[:, 2, 0]
            vor_z = gradV[:, 1, 0] - gradV[:, 0, 1]
        else:
            def get_index(ii, jj):
                return ii*3 + jj
            gradV = snap['0_VelocityGradient']
            vor_x = gradV[:, get_index(2, 1)] - gradV[:, get_index(1, 2)]
            vor_y = gradV[:, get_index(0, 2)] - gradV[:, get_index(2, 0)]
            vor_z = gradV[:, get_index(1, 0)] - gradV[:, get_index(0, 1)]
        # The vorticity vector
        # vorticity = np.stack([vor_x, vor_y, vor_z], axis=1)

        enstrophy = 0.5 * (vor_x**2 + vor_y**2 + vor_z**2)
        variable = enstrophy*snap['0_Masses']

        return variable

    def MeanMolecularWeight(snap):
        if 'GFM_Metals' in snap.info(0, False):
            hydrogen_abundance = snap['0_GFM_Metals'][:, 0]
        else:
            hydrogen_abundance = 0.76

        if 'ElectronAbundance' in snap.info(0, False):
            electron_abundance = snap['0_ElectronAbundance']
            # partially ionized
            mean_molecular_weight = 4. / (1. + 3. * hydrogen_abundance +
                                          4. * hydrogen_abundance *
                                          electron_abundance)
        else:
            # fully ionized
            mean_molecular_weight = 4. / (5. * hydrogen_abundance + 3.)
        return mean_molecular_weight

    def NumberDensity(snap):
        """
        The gas number density in cm⁻³.
        """
        from astropy import constants as c
        density = snap['0_Density'].cgs
        mean_molecular_weight = snap['0_MeanMolecularWeight']
        proton_mass = c.m_p.to('g')
        number_density_gas = density / (mean_molecular_weight * proton_mass)
        return number_density_gas

    functions = {
        "0_GFM_MetallicityTimesMasses": GFM_MetallicityTimesMasses,
        "0_Volumes": Volumes,
        "0_Temperatures": Temperatures,
        "0_Temperatures2": Temperatures2,
        "0_EnergyDissipation": EnergyDissipation,
        "0_MachnumberTimesEnergyDissipation": MachnumberTimesEnergyDissipation,
        "0_MagneticFieldSquared": MagneticFieldSquared,
        "0_MagneticFieldSquaredTimesVolumes": MagneticFieldSquaredTimesVolumes,
        "0_PressureTimesVolumes": PressureTimesVolumes,
        "0_TemperaturesTimesMasses": TemperaturesTimesMasses,
        "0_Current": Current,
        "0_Enstrophy": Enstrophy,
        "0_EnstrophyTimesMasses": EnstrophyTimesMasses,
        "0_MeanMolecularWeight": MeanMolecularWeight,
        "0_NumberDensity": NumberDensity
    }

    if info:
        return list(functions.keys())
    else:
        if variable_str in functions:
            return functions[variable_str]
        else:
            msg = ('\n\nA function to calculate the variable {} is not ' +
                   'implemented!\n\nThe currently implemented variables ' +
                   'are:\n\n{}')
            raise RuntimeError(msg.format(variable_str, functions.keys()))