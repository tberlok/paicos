import paicos as pa

aliases = {'0_Density': 'dens',
           '0_Temperatures': 'T',
           '0_MeanMolecularWeight': 'mu'}


pa.set_aliases(aliases)

snap = pa.Snapshot(pa.root_dir + '/data', 247)

try:
    snap.load_data(0, 'Density')
except RuntimeError:
    print('Expected runtime error')

snap.load_data(0, 'Masses')

snap['dens']
# snap['0_Volume']

# snap['0_Temperatures']
snap['T']

# snap.load_data(0, 'Volume')

# snap['0_Densitt']

# snap['dfadsfdsf']
