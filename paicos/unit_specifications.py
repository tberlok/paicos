# import numpy as np
import astropy.units as u

"""
Here we specify the units used in all the different fields commonly used
in Arepo simulations.

These units have been gathered by looking at the source code
and cross-referencing with the overviews at:
https://arepo-code.org/wp-content/userguide/snapshotformat.html
https://www.tng-project.org/data/docs/specifications/
"""

unit_less = u.Unit('')

Coordinates = u.Unit('arepo_length small_a / small_h')
Masses = u.Unit('arepo_mass / small_h')

# The velocity output has this scaling for historical reasons
Velocities = u.Unit('arepo_velocity  small_a^(1/2)')

# Gas velocities have a different scaling internally in Arepo,
# i.e., they are given in terms of w = u a,
# where u is the peculiar velocity
InternalVelocities = u.Unit('arepo_velocity  small_a^(-1)')

PeculiarVelocities = InternalVelocities * u.Unit('small_a')

Potential = u.Unit('arepo_velocity^2 / small_a')

MagneticField = u.Unit('arepo_pressure^(1/2) small_a^-2 small_h')

Volume = Coordinates**3
Area = Coordinates**2
Density = Masses / Volume
Density = u.Unit('arepo_density small_h2 / small_a3')

# NOTE: VelocityGradient is written out in units of
# peculiar velocities divided by internal coordinates
VelocityGradient = PeculiarVelocities / Coordinates

Pressure = u.Unit('arepo_pressure  small_h^2 small_a^-3')

default = {
    'Coordinates': Coordinates,
    'Masses': Masses,
    'Velocities': Velocities
}


StarFormationRate = u.Unit('Msun/yr')

voronoi_cells = {
    'Acceleration': PeculiarVelocities**2 / Coordinates,  # OUTPUTACCELERATION, differs from internal
    'AllowRefinement': unit_less,
    'BfieldGradient': MagneticField / Coordinates,
    'CenterOfMass': Coordinates,
    'Coordinates': Coordinates,
    'Density': Density,
    'Volume': Volume,
    'ElectronAbundance': unit_less,
    'EnergyDissipation': 'arepo_energy arepo_time^-1 small_a^-1',
    'GFM_AGNRadiation': False,  # u.Unit('erg s^-1 cm^-2')*u.Unit(str(4*np.pi)),
    'GFM_CoolingRate': 'erg cm^3 s^-1',
    'GFM_Metallicity': unit_less,
    'GFM_Metals': unit_less,
    'GFM_WindDMVelDisp': 'arepo_velocity',
    'GFM_WindHostHaloMass': Masses,
    'HighResGasMass': Masses,
    'InternalEnergy': 'arepo_energy / arepo_mass',
    'Machnumber': unit_less,
    'MagneticField': MagneticField,
    'MagneticFieldDivergence': MagneticField / Coordinates,
    'MagneticFieldDivergenceAlternative': MagneticField / Coordinates,
    'Masses': Masses,
    'NeutralHydrogenAbundance': unit_less,
    'ParticleIDs': unit_less,
    'Potential': Potential,
    'Pressure': Pressure,
    'StarFormationRate': StarFormationRate,
    'SubfindDMDensity': Density,
    'SubfindDensity': Density,
    'SubfindHsml': Coordinates,
    'SubfindVelDisp': 'arepo_velocity',
    'SoundSpeed': 'arepo_velocity',  # OUTPUT_CSND
    'Velocities': Velocities,
    'VelocityDivergence': InternalVelocities * Area / Volume,  # OUTPUT_DIVVEL
    'VelocityCurl': VelocityGradient,  # OUTPUT_CURLVEL
    'CurlVel': VelocityGradient,
    'VelocityGradient': VelocityGradient,
    'DensityGradient': Density / Coordinates,
    'PressureGradient': Pressure / Coordinates,
    'CoolingHeatingEnergy': 'arepo_energy / arepo_time',
    'SurfaceArea': Area,
    'NumFacesCell': unit_less
}

dark_matter = {
    'Coordinates': Coordinates,
    'Masses': Masses,
    'ParticleIDs': unit_less,
    'Potential': Potential,
    'SubfindDMDensity': Density,
    'SubfindDensity': Density,
    'SubfindHsml': Coordinates,
    'SubfindVelDisp': 'arepo_velocity',
    'Velocities': Velocities
}

stars = {
    'Coordinates': Coordinates,
    'GFM_InitialMass': Masses,
    'GFM_Metallicity': unit_less,
    'GFM_Metals': unit_less,
    'GFM_StellarFormationTime': unit_less,
    'GFM_StellarPhotometrics': 'mag',
    'Masses': Masses,
    'ParticleIDs': unit_less,
    'Potential': Potential,
    'SubfindDMDensity': Density,
    'SubfindDensity': Density,
    'SubfindHsml': Coordinates,
    'SubfindVelDisp': 'arepo_velocity',
    'Velocities': Velocities
}


energy_inj = u.Unit('small_a^2 small_h^-1 arepo_energy')
Mdot = u.Unit('arepo_mass / arepo_time')

black_holes = {
    'BH_CumEgyInjection_QM': energy_inj,
    'BH_CumEgyInjection_RM': energy_inj,
    'BH_CumMassGrowth_QM': Masses,
    'BH_CumMassGrowth_RM': Masses,
    'BH_Density': Density,
    'BH_HostHaloMass': Masses,
    'BH_Hsml': Coordinates,
    'BH_MPB_CumEgyHigh': energy_inj,
    'BH_MPB_CumEgyLow': energy_inj,
    'BH_Mass': Masses,
    'BH_Mass_bubbles': Masses,
    'BH_Mass_ini': Masses,
    'BH_Mdot': Mdot,
    'BH_MdotBondi': Mdot,
    'BH_MdotEddington': Mdot,
    'BH_Pressure': 'arepo_mass arepo_length^-1 arepo_time^-2',
    'BH_Progs': unit_less,
    'BH_U': 'arepo_velocity^2',
    'Coordinates': Coordinates,
    'Masses': Masses,
    'ParticleIDs': unit_less,
    'Potential': Potential,
    'SubfindDMDensity': Density,
    'SubfindDensity': Density,
    'SubfindHsml': Coordinates,
    'SubfindVelDisp': 'arepo_velocity',
    'Velocities': Velocities
}

groups = {
    'GroupBHMass': Masses,
    'GroupBHMdot': Mdot,
    'GroupCM': Coordinates,
    'GroupFirstSub': unit_less,
    'GroupGasMetalFractions': unit_less,
    'GroupGasMetallicity': unit_less,
    'GroupLen': unit_less,
    'GroupLenType': unit_less,
    'GroupMass': Masses,
    'GroupMassType': Masses,
    'GroupNsubs': unit_less,
    'GroupPos': Coordinates,
    'GroupSFR': StarFormationRate,
    'GroupStarMetalFractions': unit_less,
    'GroupStarMetallicity': unit_less,
    'GroupVel': 'arepo_velocity / small_a',  # note the a-factor!
    'GroupWindMass': Masses,
    'Group_M_Crit200': Masses,
    'Group_M_Crit500': Masses,
    'Group_M_Mean200': Masses,
    'Group_M_TopHat200': Masses,
    'Group_R_Crit200': Coordinates,
    'Group_R_Crit500': Coordinates,
    'Group_R_Mean200': Coordinates,
    'Group_R_TopHat200': Coordinates
}

subhalos = {
    'SubhaloBHMass': Masses,
    'SubhaloBHMdot': Mdot,
    'SubhaloBfldDisk': MagneticField,
    'SubhaloBfldHalo': MagneticField,
    'SubhaloCM': Coordinates,
    'SubhaloGasMetalFractions': unit_less,
    'SubhaloGasMetalFractionsHalfRad': unit_less,
    'SubhaloGasMetalFractionsMaxRad': unit_less,
    'SubhaloGasMetalFractionsSfr': unit_less,
    'SubhaloGasMetalFractionsSfrWeighted': unit_less,
    'SubhaloGasMetallicity': unit_less,
    'SubhaloGasMetallicityHalfRad': unit_less,
    'SubhaloGasMetallicityMaxRad': unit_less,
    'SubhaloGasMetallicitySfr': unit_less,
    'SubhaloGasMetallicitySfrWeighted': unit_less,
    'SubhaloGrNr': unit_less,
    'SubhaloHalfmassRad': Coordinates,
    'SubhaloHalfmassRadType': Coordinates,
    'SubhaloIDMostbound': unit_less,
    'SubhaloLen': unit_less,
    'SubhaloLenType': unit_less,
    'SubhaloMass': Masses,
    'SubhaloMassInHalfRad': Masses,
    'SubhaloMassInHalfRadType': Masses,
    'SubhaloMassInMaxRad': Masses,
    'SubhaloMassInMaxRadType': Masses,
    'SubhaloMassInRad': Masses,
    'SubhaloMassInRadType': Masses,
    'SubhaloMassType': Masses,
    'SubhaloParent': unit_less,
    'SubhaloPos': Coordinates,
    'SubhaloSFR': StarFormationRate,
    'SubhaloSFRinHalfRad': StarFormationRate,
    'SubhaloSFRinMaxRad': StarFormationRate,
    'SubhaloSFRinRad': StarFormationRate,
    'SubhaloSpin': 'arepo_length / small_h arepo_velocity',  # Check!
    'SubhaloStarMetalFractions': unit_less,
    'SubhaloStarMetalFractionsHalfRad': unit_less,
    'SubhaloStarMetalFractionsMaxRad': unit_less,
    'SubhaloStarMetallicity': unit_less,
    'SubhaloStarMetallicityHalfRad': unit_less,
    'SubhaloStarMetallicityMaxRad': unit_less,
    'SubhaloStellarPhotometrics': 'mag',
    'SubhaloStellarPhotometricsMassInRad': Masses,
    'SubhaloStellarPhotometricsRad': Coordinates,
    'SubhaloVel': 'arepo_velocity',
    'SubhaloVelDisp': 'arepo_velocity',
    'SubhaloVmax': 'arepo_velocity',
    'SubhaloVmaxRad': Velocities,
    'SubhaloWindMass': Masses
}

unit_dict = {'default': default,
             'voronoi_cells': voronoi_cells,
             'dark_matter': dark_matter,
             'stars': stars,
             'black_holes': black_holes,
             'groups': groups,
             'subhalos': subhalos
             }
