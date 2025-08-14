import paicos as pa
import numpy as np
from scipy.spatial import KDTree
from paicos.trees.bvh_cpu import BinaryTree

pa.use_units(False)
snap = pa.Snapshot(pa.data_dir, 247, basename='reduced_snap',
                   load_catalog=False)
center = np.array([398968.4, 211682.6, 629969.9])
widths = np.array([2000, 2000, 2000])

# Select a cube
index = pa.util.get_index_of_cubic_region_plus_thin_layer(
    snap['0_Coordinates'], center, widths,
    snap['0_Diameters'], snap.box)

snap = snap.select(index, parttype=0)

pos = snap['0_Coordinates']
sizes = snap['0_Diameters'] * 1e-1

# Construct a binary tree
bvh_tree = BinaryTree(pos, 1.2 * sizes)

bvh_dist, bvh_ids = bvh_tree.nearest_neighbor(pos)

# Construct a scipy kd tree
kd_tree = KDTree(pos)

kd_dist, kd_ids = kd_tree.query(pos)

np.testing.assert_array_equal(kd_ids, bvh_ids)

# Random query positions
# Limit to well inside the tree, due to holes in cutout...
pos2 = (0.1 + 0.8 * np.random.rand(10000, 3)) * widths[None, :]
pos2 = (np.random.rand(10000, 3)) * widths[None, :]
pos2 += (center - widths / 2)[None, :]

bvh_dist2, bvh_ids2 = bvh_tree.nearest_neighbor(pos2)
kd_dist2, kd_ids2 = kd_tree.query(pos2)

np.testing.assert_array_equal(kd_ids2, bvh_ids2)
pa.use_units(True)
