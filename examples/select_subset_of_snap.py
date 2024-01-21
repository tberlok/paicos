import paicos as pa
import numpy as np
pa.use_units(True)
snap = pa.Snapshot(pa.data_dir, 247)

snap['0_MagneticField']
snap['1_Coordinates']
snap['1_Masses']
del snap['1_Masses']

index = snap['0_Density'] > snap['0_Density'].unit_quantity * 1e-6
del snap['0_Density']
selected_snap = snap.select(index, parttype=0)
index, = np.where(snap['0_Density'] > snap['0_Density'].unit_quantity * 1e-6)
selected_snap2 = snap.select(index, parttype=0)
np.testing.assert_array_equal(selected_snap2['0_Density'], selected_snap['0_Density'])

np.testing.assert_array_equal(selected_snap['0_Density'], snap['0_Density'][index])

assert selected_snap['0_Temperatures'].shape == selected_snap['0_Density'].shape

dm_cut = np.ones(selected_snap['1_Coordinates'].shape[0], dtype=bool)
dm_cut[:10000] = False

further_selected_snap = selected_snap.select(dm_cut, parttype=1)
further_selected_snap['1_Masses']

K = further_selected_snap['0_Temperatures'].unit_quantity
new_index = further_selected_snap['0_Temperatures'] > 5e7 * K

further_selected_snap2 = further_selected_snap.select(new_index, parttype=0)

new_index = further_selected_snap2['0_Temperatures'] < 6.5e7 * K

further_selected_snap3 = further_selected_snap2.select(new_index, parttype=0)

further_selected_snap3['0_Volume']

further_selected_snap3['0_Velocities']
