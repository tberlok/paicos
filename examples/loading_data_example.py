import paicos as pa
from paicos import root_dir
pa.use_units(True)
snap = pa.Snapshot(root_dir + '/data', 247)

snap.load_data(0, 'Density')
snap.load_data(0, 'Masses')

snap['0_MagneticField']
snap['0_Volume']
snap['0_Temperatures']
# snap.load_data(0, 'Volume')

# snap['0_Densitt']

# snap['dfadsfdsf']
