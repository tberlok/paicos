import numpy as np
import paicos as pa
import numba

def get_octant_index(morton_code: int, level: int) -> int:
    """
    Return the octant index (0–7) at a given level from a Morton code.

    Parameters:
    - morton_code: full 64-bit Morton code
    - level: the current level in the tree (0 is root)
    - max_level: total tree depth (default 21)

    Returns:
    - integer from 0 to 7 (octant index at this level)
    """
    shift = 3 * (L - level)  # 0-based indexing
    return (morton_code >> shift) & 7

L = 21

# Just a test of how node indexing works
if False:
    pos = np.ones((9, 3), dtype=np.uint64)
    pos[:] = 2**L / 2
    ll = 1
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                pos[ll, 0] = 2**L/4 * (1 + 2 * ii)
                pos[ll, 1] = 2**L/4 * (1 + 2 * jj)
                pos[ll, 2] = 2**L/4 * (1 + 2 * kk)
                ll += 1

    morton_codes = pa.trees.bvh_cpu.get_morton_keys64_old(pos)
    for ii in range(1, 9):
        print(get_octant_index(morton_codes[ii], 1))


def width_of_node(level):
    return 2**L / 2**l

@numba.jit()
def part1by2_64(n):
    n &= 0x1fffff
    n = (n | (n << 32)) & 0x1f00000000ffff
    n = (n | (n << 16)) & 0x1f0000ff0000ff
    n = (n | (n << 8)) & 0x100f00f00f00f00f
    n = (n | (n << 4)) & 0x10c30c30c30c30c3
    n = (n | (n << 2)) & 0x1249249249249249
    return n


@numba.jit()
def unpart1by2_64(n):
    n = numba.uint64(n)
    n &= 0x1249249249249249
    n = (n ^ (n >> 2)) & 0x10c30c30c30c30c3
    n = (n ^ (n >> 4)) & 0x100f00f00f00f00f
    n = (n ^ (n >> 8)) & 0x1f0000ff0000ff
    n = (n ^ (n >> 16)) & 0x1f00000000ffff
    n = (n ^ (n >> 32)) & 0x1fffff
    return n

@numba.jit()
def encode64(pos):
    xx = part1by2_64(pos[0])
    yy = part1by2_64(pos[1])
    zz = part1by2_64(pos[2])
    return xx * 4 + yy * 2 + zz

@numba.jit()
def decode64(key):
    x = unpart1by2_64(key >> 2)
    y = unpart1by2_64(key >> 1)
    z = unpart1by2_64(key)
    return numba.float64(x), numba.float64(y), numba.float64(z)




import bisect

# tree_dict = {1: {index: np.sum(indices == index) for index in range(8)}}
# each node needs to store information about
# first index to particle position array, number of particles contained, index to first child in node array,
# the number of children, index to parent in node array, the morton key for the node position.
# and the level at which the node resides. Position of the node in the full domain and the width of node
# can then be calculated from the level and the morton key.
# Use a queue to process the nodes as we resolve them.
# We can in principle introduce constraints such as making a dense tree or even a restricted tree.
# For a restricted tree I guess we would first need a pass to decide which levels to refine based on particle counts.
# After that, we would force refine the neighbours which are not sufficiently refined. It appears to be uncommon to have
# to make further passes where the neigbours of the neigbors are forced to refine. But that is in principle what is
# needed. For a restricted tree we would always have 8 children, and so the layout that I imagine above might not be
# the best (The number of children array would just contain 8.). Btw, the number of children can be stored as an uint8 (0, 255).
# Let's also add an array for whether to refine.

Ncrit = 8
dense_tree = False

np.random.seed(4)
N = 10**6
pos = np.random.rand(N, 3) * 2**L
pos_uint = pos.astype(np.uint64)
pos_morton = pa.trees.bvh_cpu.get_morton_keys64_old(pos_uint)
sort_index = np.argsort(pos_morton)
pos_uint = pos_uint[sort_index]
pos = pos[sort_index]
pos_morton = pos_morton[sort_index]

node_morton = np.empty(2*N, dtype=np.uint64)
level = np.empty(2*N, dtype=np.uint8)
num_children = np.zeros(2*N, dtype=np.uint8)
num_points = np.empty(2*N, dtype=np.int32)
first_point_index = np.empty(2*N, dtype=np.int32)
first_child_index = np.empty(2*N, dtype=np.int32)
parent_index = np.empty(2*N, dtype=np.int32)
needs_refinement = np.zeros(2*N, dtype=np.bool)

offsets = []
for ii in range(2):
    for jj in range(2):
        for kk in range(2):
            offsets.append(np.array([-1 + 2*ii, -1 + 2*jj, -1 + 2*kk]))


# Set root node
num_nodes = np.int32(1)
node_id = np.int32(0)
node_morton[node_id] = encode64(np.ones(3, dtype=np.uint64) * 2**L // 2)
num_points[node_id] = pos.shape[0]
first_point_index[node_id] = 0
first_child_index[node_id]
parent_index[node_id] = -1
level[node_id] = 0

counts = [0] * 8

def refine_node(node_id, num_nodes):
    if needs_refinement[node_id]:
        if num_children[node_id] > 0:
            # already refined, just return
            return num_nodes
        new_level = level[node_id] + 1
        assert new_level <= L, f"new_level={new_level} exceeds allowed maximum level of 21"
        start_index = first_point_index[node_id]
        end_index = start_index + num_points[node_id]

        indices = get_octant_index(pos_morton[start_index:end_index], new_level)
        start_indices = [bisect.bisect_left(indices, octal_index) for octal_index in range(8)]

        counts = np.diff(start_indices + [num_points[node_id]])

        node_pos = decode64(node_morton[node_id])
        new_node_width = 2**L // 2**np.uint64(new_level)
        # Adding children
        first_child_set = False
        new_node_id = num_nodes - 1
        for ii in range(8):
            if counts[ii] > 0 or dense_tree:
                new_node_id += 1
                num_children[node_id] += 1
                if not first_child_set:
                    first_child_index[node_id] = new_node_id
                    first_child_set = True

                new_pos = node_pos + offsets[ii] * new_node_width // 2

                node_morton[new_node_id] = encode64(new_pos.astype(np.uint64))
                num_points[new_node_id] = counts[ii]
                first_point_index[new_node_id] = start_index + start_indices[ii]
                parent_index[new_node_id] = node_id
                level[new_node_id] = new_level
        num_nodes = num_nodes + num_children[node_id]
    else:
        print(f'node_id {node_id} does not need refinement')

    return num_nodes

# Build tree
while node_id < num_nodes:
    # print(node_id, num_nodes)
    if num_points[node_id] > Ncrit:
        needs_refinement[node_id] = True
    else:
        needs_refinement[node_id] = False

    if needs_refinement[node_id]:
        num_nodes = refine_node(node_id, num_nodes)
    node_id += 1

node_morton = node_morton[:num_nodes]
level = level[:num_nodes]
num_children = num_children[:num_nodes]
num_points = num_points[:num_nodes]
first_point_index = first_point_index[:num_nodes]
first_child_index = first_child_index[:num_nodes]
parent_index = parent_index[:num_nodes]
needs_refinement = needs_refinement[:num_nodes]

# That's it, although we need to propagate bounds AABBs upwards for a BVH.
# These are the leaf nodes
num_points[num_children==0]
