"""
Functions for getting derived variables of gas
"""
import numpy as np

# pylint: disable=import-outside-toplevel


def GFM_MetallicityTimesMasses(snap, get_dependencies=False):
    """Returns the metallicity times the mass."""
    if get_dependencies:
        return ['0_Masses', '0_GFM_Metallicity']
    return snap['0_Masses'] * snap['0_GFM_Metallicity']


def Volume(snap, get_dependencies=False):
    """Returns the volume of the Voronoi cells"""
    if get_dependencies:
        return ['0_Masses', '0_Density']
    return snap["0_Masses"] / snap["0_Density"]


def MachnumberTimesEnergyDissipation(snap, get_dependencies=False):
    """Returns the Mach number times the energy dissipation."""
    if get_dependencies:
        return ['0_Machnumber', '0_EnergyDissipation']
    variable = snap['0_Machnumber'] * snap['0_EnergyDissipation']
    return variable


def MagneticFieldSquared(snap, get_dependencies=False):
    """Returns the magnetic field strength squared."""
    if get_dependencies:
        return ['0_MagneticField']
    return np.sum(snap['0_MagneticField']**2, axis=1)


def MagneticFieldStrength(snap, get_dependencies=False):
    """Returns the magnetic field strength."""
    if get_dependencies:
        return ['0_MagneticField']
    return np.sqrt(np.sum(snap['0_MagneticField']**2, axis=1))


def VelocityMagnitude(snap, get_dependencies=False):
    """Returns the magnitude of the gas velocity."""
    if get_dependencies:
        return ['0_Velocities']
    return np.sqrt(np.sum(snap['0_Velocities']**2, axis=1))


def MagneticFieldSquaredTimesVolume(snap, get_dependencies=False):
    """Returns B² × V """
    if get_dependencies:
        return ['0_Volume', '0_MagneticField']
    variable = snap["0_Volume"] * np.sum(snap['0_MagneticField']**2, axis=1)
    return variable


def Pressure(snap, get_dependencies=False):
    """Returns the gas pressures"""
    if get_dependencies:
        return ['0_InternalEnergy', '0_Density']

    if snap.gamma == 1:
        msg = 'Temperature field not supported for isothermal EOS!'
        raise RuntimeError(msg)
    gm1 = snap.gamma - 1
    variable = snap["0_InternalEnergy"] * snap["0_Density"] * gm1
    return variable.to('arepo_pressure')


def PressureTimesVolume(snap, get_dependencies=False):
    """Returns the gas pressure times the volume."""
    if get_dependencies:
        return ['0_Pressure', '0_Volume']

    if '0_Pressure' in snap:
        return snap['0_Pressure'] * snap['0_Volume']

    if snap.gamma != 1:
        gm1 = snap.gamma - 1
        variable = snap["0_Masses"] * snap["0_InternalEnergy"] * gm1
    else:
        variable = snap['0_Volume'] * snap['0_Pressure']

    return variable


def Temperatures(snap, get_dependencies=False):
    """Returns the temperature in Kelvin."""

    if get_dependencies:
        return ['0_InternalEnergy', '0_MeanMolecularWeight']

    from astropy import constants as c
    mhydrogen = c.m_e + c.m_p

    gm1 = snap.gamma - 1

    if snap.gamma == 1:
        msg = 'Temperature field not supported for isothermal EOS!'
        raise RuntimeError(msg)

    mmean = snap['0_MeanMolecularWeight']

    # temperature in Kelvin
    from .. import settings
    if settings.use_units:
        variable = (gm1 * snap["0_InternalEnergy"]
                    * mmean * mhydrogen).to('K')
    else:
        u_v = snap.arepo_units['unit_velocity']
        variable = (gm1 * snap["0_InternalEnergy"]
                    * u_v**2 * mmean * mhydrogen
                    ).to('K').value
    return variable


def TemperaturesTimesMasses(snap, get_dependencies=False):
    """Returns the temperature times masses."""
    if get_dependencies:
        return ['0_Temperatures', '0_Masses']

    return snap["0_Temperatures"] * snap['0_Masses']


def Current(snap, get_dependencies=False):
    """Returns the magnitude of the current, i.e., |∇×B|."""

    if get_dependencies:
        return ['0_BfieldGradient']

    def get_index(ii, jj):
        return ii * 3 + jj
    gradB = snap['0_BfieldGradient']
    J_x = gradB[:, get_index(2, 1)] - gradB[:, get_index(1, 2)]
    J_y = gradB[:, get_index(0, 2)] - gradB[:, get_index(2, 0)]
    J_z = gradB[:, get_index(1, 0)] - gradB[:, get_index(0, 1)]

    J = np.sqrt(J_x**2 + J_y**2 + J_z**2)
    return J


def Enstrophy(snap, get_dependencies=False):
    """Returns the enstrophy, i.e., 1/2|∇×v|²."""

    if get_dependencies:
        return ['0_VelocityGradient']

    def get_index(ii, jj):
        return ii * 3 + jj
    gradV = snap['0_VelocityGradient'][()]
    vor_x = gradV[:, get_index(2, 1)] - gradV[:, get_index(1, 2)]
    vor_y = gradV[:, get_index(0, 2)] - gradV[:, get_index(2, 0)]
    vor_z = gradV[:, get_index(1, 0)] - gradV[:, get_index(0, 1)]

    enstrophy = 0.5 * (vor_x**2 + vor_y**2 + vor_z**2)
    return enstrophy


def EnstrophyTimesMasses(snap, get_dependencies=False):
    """ Returns the enstrophy times masses."""

    if get_dependencies:
        return ['0_VelocityGradient']

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
            return ii * 3 + jj
        gradV = snap['0_VelocityGradient']
        vor_x = gradV[:, get_index(2, 1)] - gradV[:, get_index(1, 2)]
        vor_y = gradV[:, get_index(0, 2)] - gradV[:, get_index(2, 0)]
        vor_z = gradV[:, get_index(1, 0)] - gradV[:, get_index(0, 1)]
    # The vorticity vector
    # vorticity = np.stack([vor_x, vor_y, vor_z], axis=1)

    enstrophy = 0.5 * (vor_x**2 + vor_y**2 + vor_z**2)
    variable = enstrophy * snap['0_Masses']

    return variable


def MeanMolecularWeight(snap, get_dependencies=False):
    """Returns the mean molecular weight, μ."""

    if get_dependencies:
        return ['0_Density']

    if '0_GFM_Metals' in snap.info(0, False):
        hydrogen_abundance = snap['0_GFM_Metals'][:, 0]
    else:
        hydrogen_abundance = np.array([0.76])

    if '0_ElectronAbundance' in snap.info(0, False):
        electron_abundance = snap['0_ElectronAbundance']
        # partially ionized
        mean_molecular_weight = 4. / (1. + 3. * hydrogen_abundance
                                      + 4. * hydrogen_abundance
                                      * electron_abundance)
    else:
        # fully ionized
        mean_molecular_weight = 4. / (5. * hydrogen_abundance + 3.)
    return mean_molecular_weight


def NumberDensity(snap, get_dependencies=False):
    """
    The gas number density in cm⁻³.
    """
    if get_dependencies:
        return ['0_Density']

    from astropy import constants as c
    density = snap['0_Density'].cgs
    mean_molecular_weight = snap['0_MeanMolecularWeight']
    proton_mass = c.m_p.to('g')
    number_density_gas = density / (mean_molecular_weight * proton_mass)
    return number_density_gas


def MagneticCurvature(snap, get_dependencies=False):
    """Returns the length of the magnetic curvature vector."""

    if get_dependencies:
        return ['0_MagneticField', '0_BfieldGradient']

    from .. import util

    @util.remove_astro_units
    def get_func(B, gradB):
        from ..cython.get_derived_variables import get_curvature
        return get_curvature(B, gradB)

    curva = get_func(snap['0_MagneticField'], snap['0_BfieldGradient'])
    unit_quantity = snap['0_BfieldGradient'].uq / snap['0_MagneticField'].uq
    curva = curva * unit_quantity
    return curva


def VelocityCurvature(snap, get_dependencies=False):
    """Returns the length of the velocity curvature vector."""
    if get_dependencies:
        return ['0_Velocities', '0_VelocityGradient']

    from .. import util

    @util.remove_astro_units
    def get_func(V, gradV):
        from ..cython.get_derived_variables import get_curvature
        return get_curvature(V, gradV)

    curva = get_func(snap['0_Velocities'], snap['0_VelocityGradient'])
    unit_quantity = snap['0_VelocityGradient'].uq / snap['0_Velocities'].uq
    curva = curva * unit_quantity
    return curva


def Diameters(snap, get_dependencies=False):
    """Returns the cell diameters."""
    if get_dependencies:
        return ['0_Volume']

    return 2 * np.cbrt((snap['0_Volume']) / (4.0 * np.pi / 3.0))


functions = {
    "0_GFM_MetallicityTimesMasses": GFM_MetallicityTimesMasses,
    "0_Volume": Volume,
    "0_Temperatures": Temperatures,
    "0_MachnumberTimesEnergyDissipation": MachnumberTimesEnergyDissipation,
    "0_MagneticFieldSquared": MagneticFieldSquared,
    "0_MagneticFieldStrength": MagneticFieldStrength,
    "0_MagneticFieldSquaredTimesVolume": MagneticFieldSquaredTimesVolume,
    "0_Pressure": Pressure,
    "0_PressureTimesVolume": PressureTimesVolume,
    "0_TemperaturesTimesMasses": TemperaturesTimesMasses,
    "0_Current": Current,
    "0_Enstrophy": Enstrophy,
    "0_EnstrophyTimesMasses": EnstrophyTimesMasses,
    "0_MeanMolecularWeight": MeanMolecularWeight,
    "0_NumberDensity": NumberDensity,
    "0_MagneticCurvature": MagneticCurvature,
    "0_VelocityMagnitude": VelocityMagnitude,
    "0_VelocityCurvature": VelocityCurvature,
    "0_Diameters": Diameters
}
