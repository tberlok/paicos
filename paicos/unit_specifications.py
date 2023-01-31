# import numpy as np
import astropy.units as u

unit_less = u.Unit('')

Coordinates = u.Unit('arepo_length small_a / small_h')
Masses = u.Unit('arepo_mass / small_h')
Velocities = u.Unit('arepo_mass / small_h')

Potential = u.Unit('arepo_velocity^2 / small_a')

MagneticField = u.Unit('arepo_pressure^(1/2) small_a^-2 small_h')

Volume = Coordinates**3
Density = Masses / Volume
Density = u.Unit('arepo_density small_h2 / small_a3')

default = {
    'Coordinates': Coordinates,
    'Masses': Masses,
    'Velocities': Velocities
}

voronoi_cells = {
    'AllowRefinement': unit_less,
    'BfieldGradient': MagneticField / Coordinates,
    'CenterOfMass': Coordinates,
    'Coordinates': Coordinates,
    'Density': Density,
    'ElectronAbundance': unit_less,
    'EnergyDissipation': False,
    # (1/𝑎)1010𝑀⊙/ckpc(km/s)3
    'GFM_AGNRadiation': False,  # u.Unit('erg s^-1 cm^-2')*u.Unit(str(4*np.pi)),
    'GFM_CoolingRate': u.Unit('erg cm^3 s^-1'),
    'GFM_Metallicity': unit_less,
    'GFM_Metals': unit_less,
    'GFM_WindDMVelDisp': u.Unit('arepo_velocity'),
    'GFM_WindHostHaloMass': Masses,
    'HighResGasMass': Masses,
    'InternalEnergy': u.Unit('arepo_energy / arepo_mass'),
    'Machnumber': unit_less,
    'MagneticField': MagneticField,
    'MagneticFieldDivergence': MagneticField / Coordinates,
    'MagneticFieldDivergenceAlternative': MagneticField / Coordinates,
    'Masses': Masses,
    'NeutralHydrogenAbundance': unit_less,
    'ParticleIDs': unit_less,
    'Potential': Potential,
    'Pressure': u.Unit('arepo_pressure  small_h^2 small_a^-3'),
    'StarFormationRate': u.Unit('Msun yr^-1'),
    'SubfindDMDensity': Density,
    'SubfindDensity': Density,
    'SubfindHsml': Coordinates,
    'SubfindVelDisp': u.Unit('arepo_velocity'),
    'Velocities': Velocities,
    'VelocityGradient': Velocities/Coordinates
}

dark_matter = {
    'Coordinates': Coordinates,
    'Masses': Masses,
    'ParticleIDs': unit_less,
    'Potential': Potential,
    'SubfindDMDensity': Density,
    'SubfindDensity': Density,
    'SubfindHsml': Coordinates,
    'SubfindVelDisp': u.Unit('arepo_velocity'),
    'Velocities': Velocities
}

stars = {
    'Coordinates': Coordinates,
    'GFM_InitialMass': Masses,
    'GFM_Metallicity': unit_less,
    'GFM_Metals': unit_less,
    'GFM_StellarFormationTime': unit_less,
    'GFM_StellarPhotometrics': False,
    'Masses': Masses,
    'ParticleIDs': unit_less,
    'Potential': Potential,
    'SubfindDMDensity': Density,
    'SubfindDensity': Density,
    'SubfindHsml': Coordinates,
    'SubfindVelDisp': u.Unit('arepo_velocity'),
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
    'BH_Pressure': u.Unit('arepo_mass arepo_length^-1 arepo_time^-2'),
    'BH_Progs': unit_less,
    'BH_U': u.Unit('arepo_velocity^2'),
    'Coordinates': Coordinates,
    'Masses': Masses,
    'ParticleIDs': unit_less,
    'Potential': Potential,
    'SubfindDMDensity': Density,
    'SubfindDensity': Density,
    'SubfindHsml': Coordinates,
    'SubfindVelDisp': u.Unit('arepo_velocity'),
    'Velocities': Velocities
}

groups = {
    'GroupBHMass': Masses,
    'GroupBHMdot': Mdot,
    'GroupCM': Coordinates,
    'GroupFirstSub': False,
    'GroupGasMetalFractions': False,
    'GroupGasMetallicity': False,
    'GroupLen': False,
    'GroupLenType': False,
    'GroupMass': Masses,
    'GroupMassType': False,
    'GroupNsubs': False,
    'GroupPos': Coordinates,
    'GroupSFR': False,
    'GroupStarMetalFractions': False,
    'GroupStarMetallicity': False,
    'GroupVel': False,
    'GroupWindMass': False,
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
    'SubhaloBHMass': False,
    'SubhaloBHMdot': False,
    'SubhaloBfldDisk': False,
    'SubhaloBfldHalo': False,
    'SubhaloCM': False,
    'SubhaloGasMetalFractions': False,
    'SubhaloGasMetalFractionsHalfRad': False,
    'SubhaloGasMetalFractionsMaxRad': False,
    'SubhaloGasMetalFractionsSfr': False,
    'SubhaloGasMetalFractionsSfrWeighted': False,
    'SubhaloGasMetallicity': False,
    'SubhaloGasMetallicityHalfRad': False,
    'SubhaloGasMetallicityMaxRad': False,
    'SubhaloGasMetallicitySfr': False,
    'SubhaloGasMetallicitySfrWeighted': False,
    'SubhaloGrNr': False,
    'SubhaloHalfmassRad': False,
    'SubhaloHalfmassRadType': False,
    'SubhaloIDMostbound': False,
    'SubhaloLen': False,
    'SubhaloLenType': False,
    'SubhaloMass': False,
    'SubhaloMassInHalfRad': False,
    'SubhaloMassInHalfRadType': False,
    'SubhaloMassInMaxRad': False,
    'SubhaloMassInMaxRadType': False,
    'SubhaloMassInRad': False,
    'SubhaloMassInRadType': False,
    'SubhaloMassType': False,
    'SubhaloParent': False,
    'SubhaloPos': False,
    'SubhaloSFR': False,
    'SubhaloSFRinHalfRad': False,
    'SubhaloSFRinMaxRad': False,
    'SubhaloSFRinRad': False,
    'SubhaloSpin': False,
    'SubhaloStarMetalFractions': False,
    'SubhaloStarMetalFractionsHalfRad': False,
    'SubhaloStarMetalFractionsMaxRad': False,
    'SubhaloStarMetallicity': False,
    'SubhaloStarMetallicityHalfRad': False,
    'SubhaloStarMetallicityMaxRad': False,
    'SubhaloStellarPhotometrics': False,
    'SubhaloStellarPhotometricsMassInRad': False,
    'SubhaloStellarPhotometricsRad': False,
    'SubhaloVel': False,
    'SubhaloVelDisp': False,
    'SubhaloVmax': False,
    'SubhaloVmaxRad': False,
    'SubhaloWindMass': False
}
