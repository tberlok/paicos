import paicos as pa
import numpy as np
from scipy.spatial import KDTree
from paicos.trees.bvh_cpu import BinaryTree
from paicos.trees.bvh_cpu import get_morton_keys64_old, get_morton_keys64
from paicos.trees.bvh_cpu import propagate_bounds_upwards_old, propagate_bounds_upwards
from paicos.trees.bvh_cpu import generateHierarchy_old, generateHierarchy
from paicos.trees.bvh_cpu import set_leaf_bounding_volumes_old, set_leaf_bounding_volumes

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
sizes = snap['0_Diameters']

L = 21

positions = pos
sizes = 1.2 * sizes

_pos = np.array(positions)

# Convert _position to units from [0, 1) * 2**L
off_sets = np.min(_pos, axis=0)
_pos -= off_sets[None, :]

max_pos = np.max(_pos)
conversion_factor = (2**L - 1) / max_pos
_pos *= conversion_factor

# if use_units take the value...
# pos = pos.value
#

# Calculate uint64 and Morton keys
_pos_uint = _pos.astype(np.uint64)

morton_keys_old = get_morton_keys64_old(_pos_uint)
morton_keys = get_morton_keys64(_pos_uint)

np.testing.assert_array_equal(morton_keys, morton_keys_old)
# print(morton_keys)

# Store index for going back to original ids.
sort_index = np.argsort(morton_keys)

# Do the sorting
morton_keys = morton_keys[sort_index]
_pos = _pos[sort_index, :]
_pos_uint = _pos_uint[sort_index, :]

# # Some tree properties
num_leafs = morton_keys.shape[0]
num_internal_nodes = num_leafs - 1
num_leafs_and_nodes = 2 * num_leafs - 1

# Allocate arrays
children = -1 * np.ones((num_internal_nodes, 2), dtype=int)
parents = -1 * np.ones(num_leafs_and_nodes, dtype=int)

children_old = -1 * np.ones((num_internal_nodes, 2), dtype=int)
parents_old = -1 * np.ones(num_leafs_and_nodes, dtype=int)

# # This sets the parent and children properties
generateHierarchy(morton_keys, children, parents)
generateHierarchy_old(morton_keys, children_old, parents_old)

np.testing.assert_array_equal(parents, parents_old)
np.testing.assert_array_equal(children, children_old)

# Set the boundaries for the leafs
bounds_old = np.zeros((num_leafs_and_nodes, 3, 2), dtype=np.uint64)
bounds = np.zeros((num_leafs_and_nodes, 3, 2), dtype=np.uint64)
bounds_cython = np.zeros((num_leafs_and_nodes, 3, 2), dtype=np.uint64)

set_leaf_bounding_volumes_old(bounds_old, _pos,
                              sizes[sort_index],
                              conversion_factor)
set_leaf_bounding_volumes(bounds, _pos,
                          sizes[sort_index],
                          conversion_factor, L)

np.testing.assert_array_equal(bounds, bounds_old)

propagate_bounds_upwards_old(bounds_old, parents, children)
propagate_bounds_upwards(bounds, parents, children)

np.testing.assert_array_equal(bounds, bounds_old)
