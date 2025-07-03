import paicos as pa
import numpy as np
from scipy.spatial import KDTree
pa.use_units(False)
from paicos.trees.bvh_cpu import BinaryTree
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
sizes = snap['0_Diameters']

# Construct a binary tree
bvh_tree = BinaryTree(pos, 1.2 * sizes)

bvh_dist, bvh_ids = bvh_tree.nearest_neighbor(pos)

radius = 100.

indices, num_neighbours = bvh_tree.range_search(pos, radius, max_neighbors=10)

indices, num_neighbours = bvh_tree.range_search(pos, radius, max_neighbors=1000)

brute_force_indices = []
for ii in range(pos.shape[0]):
    brute_force_indices.append(np.arange(pos.shape[0])[pa.util.get_index_of_radial_range(pos, pos[ii, :], 0, radius)])
    np.testing.assert_array_equal(np.sort(indices[ii]), brute_force_indices[ii])


