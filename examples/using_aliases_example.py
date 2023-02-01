import paicos as pa

aliases = {'0_Temperatures': 'T',
           '0_MeanMolecularWeight': 'mu',
           '0_Masses': 'mass',
           '0_InternalEnergy': 'u',
           '0_Density': 'rho',
           '0_MagneticField': 'bfld',
           '0_Velocities': 'vel',
           '0_Pressure': 'pres'}


pa.set_aliases(aliases)

p = pa.Snapshot(pa.root_dir + '/data', 247)
p['rho']
p['T']
p['bfld']

p.info(0)
