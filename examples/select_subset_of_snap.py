import paicos as pa
from paicos import root_dir
import numpy as np
pa.use_units(True)
snap = pa.Snapshot(root_dir + '/data', 247)

snap['0_MagneticField']
snap['1_Coordinates']
index = snap['0_Density'] > snap['0_Density'].unit_quantity*1e-6
selected_snap = snap.select(index)

assert selected_snap['0_Temperatures'].shape == selected_snap['0_Density'].shape

dm_cut = np.ones(selected_snap['1_Coordinates'].shape[0], dtype=bool)
dm_cut[:10000] = False

further_selected_snap = selected_snap.select(dm_cut, 1)

K = further_selected_snap['0_Temperatures'].unit_quantity
new_index = further_selected_snap['0_Temperatures'] > 5e7 * K

further_selected_snap2 = further_selected_snap.select(new_index)

new_index = further_selected_snap2['0_Temperatures'] < 6.5e7 * K

further_selected_snap3 = further_selected_snap2.select(new_index)

further_selected_snap3['0_Volumes']

further_selected_snap3['0_Velocities']
