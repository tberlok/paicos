import paicos as pa

snap = pa.Snapshot(pa.root_dir + '/data', 247)
center = snap.Cat.Group['GroupPos'][0]

widths = [2000, 2000, 2000]

# Use OpenMP parallel Cython code
pos = snap['0_Coordinates']

snap['0_Density']
snap['0_Masses']
# snap['0_Velocities']


index = pa.util.get_index_of_cubic_region(pos, center, widths, snap.box)

snap = snap.select(index)

snap.save_new_snapshot('reduced_snap')
