import paicos as pa
from paicos import root_dir
pa.use_units(True)
snap = pa.Snapshot(root_dir + '/data', 247)

snap['0_MagneticField']
index = snap['0_Density'] > snap['0_Density'].unit_quantity*1e-6
selected_snap = snap.select(index)

assert selected_snap['0_Temperatures'].shape == selected_snap['0_Density'].shape
