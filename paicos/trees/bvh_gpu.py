import numba
from numba import cuda
import math
import cupy as cp

# Hardcoded for uint64 coordinates (don't change without also changing morton
# code functions)
L = 21


@cuda.jit(device=True, inline=True)
def part1by2_64(n):
    n &= 0x1fffff
    n = (n | (n << 32)) & 0x1f00000000ffff
    n = (n | (n << 16)) & 0x1f0000ff0000ff
    n = (n | (n << 8)) & 0x100f00f00f00f00f
    n = (n | (n << 4)) & 0x10c30c30c30c30c3
    n = (n | (n << 2)) & 0x1249249249249249
    return n


@cuda.jit(device=True, inline=True)
def unpart1by2_64(n):
    n &= 0x1249249249249249
    n = (n ^ (n >> 2)) & 0x10c30c30c30c30c3
    n = (n ^ (n >> 4)) & 0x100f00f00f00f00f
    n = (n ^ (n >> 8)) & 0x1f0000ff0000ff
    n = (n ^ (n >> 16)) & 0x1f00000000ffff
    n = (n ^ (n >> 32)) & 0x1fffff
    return n


@cuda.jit(device=True, inline=True)
def encode64(x, y, z):
    xx = part1by2_64(x)
    yy = part1by2_64(y)
    zz = part1by2_64(z)
    return xx * 4 + yy * 2 + zz


@cuda.jit(device=True, inline=True)
def decode64(key):
    x = unpart1by2_64(key >> 2)
    y = unpart1by2_64(key >> 1)
    z = unpart1by2_64(key)
    return numba.float64(x), numba.float64(y), numba.float64(z)


@cuda.jit
def get_morton_keys64(pos, morton_keys):
    ip = cuda.grid(1)
    if ip < pos.shape[0]:
        morton_keys[ip] = encode64(pos[ip, 0], pos[ip, 1], pos[ip, 2])


@cuda.jit
def decode_morton_keys64(morton_keys, x, y, z):
    ip = cuda.grid(1)
    if ip < x.shape[0]:
        x[ip], y[ip], z[ip] = decode64(morton_keys[ip])


@cuda.jit(device=True, inline=True)
def device_clf_64(tmp):
    # This is
    # https://docs.nvidia.com/cuda/'libdevice-users-guide/__nv_clzll.html#__nv_clzll
    return cuda.libdevice.clzll(tmp)


@cuda.jit(device=True, inline=True)
def delta(sortedMortonCodes, i, j):
    codeA = sortedMortonCodes[i]
    codeB = sortedMortonCodes[j]

    if codeA == codeB:
        # Fallback: compare indices instead of codes
        return 64 + cuda.libdevice.clzll(numba.uint64(i ^ j))
    else:
        return cuda.libdevice.clzll(codeA ^ codeB)


@cuda.jit(device=True, inline=True)
def findSplit(sortedMortonCodes, first, last):
    # Calculate the number of highest bits that are the same
    # for all objects, using the count-leading-zeros intrinsic.

    commonPrefix = delta(sortedMortonCodes, first, last)

    # Use binary search to find where the next bit differs.
    # Specifically, we are looking for the highest object that
    # shares more than commonPrefix bits with the first one.

    split = first  # initial guess
    step = last - first

    while (step > 1):
        step = (step + 1) >> 1  # exponential decrease
        newSplit = split + step  # proposed new position

        if (newSplit < last):
            splitPrefix = delta(sortedMortonCodes, first, newSplit)
            if (splitPrefix > commonPrefix):
                split = newSplit  # accept proposal

    return split


@cuda.jit()
def generateHierarchy(sortedMortonCodes, tree_children, tree_parents):
    """
    Generate tree using the sorted Morton code.
    This is done in parallel, see this blogpost
    https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    and the corresponding paper:

    """
    n_codes = sortedMortonCodes.shape[0]
    num_internal_nodes = n_codes - 1
    idx = cuda.grid(1)

    # Launch thread for each internal node
    if idx < num_internal_nodes:

        # Find out which range of objects the node corresponds to.
        # (This is where the magic happens!)

        # Determine range
        first, last = determineRange(sortedMortonCodes, n_codes, idx)

        # Determine where to split the range.
        split = findSplit(sortedMortonCodes, first, last)

        # Select childA.
        if (split == first):
            # Is a leaf
            childA = split + num_internal_nodes
        else:
            # Is in an internal node
            childA = split

        # Select childB.
        if (split + 1 == last):
            childB = split + 1 + num_internal_nodes
        else:
            childB = split + 1

        # Record parent-child relationships.
        tree_children[idx, 0] = childA
        tree_children[idx, 1] = childB
        tree_parents[childA] = idx
        tree_parents[childB] = idx
    cuda.syncthreads()


@cuda.jit(device=True, inline=True)
def determineRange(sortedMortonCodes, n_codes, idx):
    """
    The determine range function needed by the generateHierarchy function.
    This code has has been adapted from the CornerStone CUDA/C++ code,
    see constructInternalNode in btree.hpp.
    """
    d = 1
    minPrefixLength = -1
    firstIndex = idx
    max_last = n_codes

    if (firstIndex > 0):
        # Count leading zeros
        lz_p1 = delta(sortedMortonCodes, firstIndex, firstIndex + 1)
        lz_m1 = delta(sortedMortonCodes, firstIndex, firstIndex - 1)

        if lz_p1 > lz_m1:
            d = 1
            minPrefixLength = lz_m1
        else:
            d = -1
            minPrefixLength = lz_p1

    searchRange = 2
    secondIndex = firstIndex + searchRange * d

    while (0 <= secondIndex and secondIndex < max_last):
        lz_first_second = delta(sortedMortonCodes, firstIndex, secondIndex)
        if lz_first_second > minPrefixLength:
            searchRange *= 2
            secondIndex = firstIndex + searchRange * d
            if secondIndex < max_last:
                lz_first_second = delta(sortedMortonCodes, firstIndex, secondIndex)
            else:
                break
        else:
            break

    # start binary search with known searchRange
    secondIndex = firstIndex
    while searchRange > 1:
        searchRange = (searchRange + 1) // 2
        newJdx = secondIndex + searchRange * d
        if (0 <= newJdx and newJdx < max_last):
            lz_f_Jdx = delta(sortedMortonCodes, firstIndex, newJdx)
            if lz_f_Jdx > minPrefixLength:
                secondIndex = newJdx

    first = min(secondIndex, firstIndex)
    last = max(secondIndex, firstIndex)

    return first, last


@cuda.jit()
def set_leaf_bounding_volumes(tree_bounds, points, half_size, conversion_factor):
    num_internal_nodes = half_size.shape[0] - 1
    num_leafs = half_size.shape[0]
    f = conversion_factor
    ip = cuda.grid(1)
    if ip < num_leafs:
        x_min = max(math.floor((points[ip, 0] - f * half_size[ip])), 0)
        y_min = max(math.floor((points[ip, 1] - f * half_size[ip])), 0)
        z_min = max(math.floor((points[ip, 2] - f * half_size[ip])), 0)
        x_max = min(math.ceil((points[ip, 0] + f * half_size[ip])), 2**L - 1)
        y_max = min(math.ceil((points[ip, 1] + f * half_size[ip])), 2**L - 1)
        z_max = min(math.ceil((points[ip, 2] + f * half_size[ip])), 2**L - 1)
        tree_bounds[ip + num_internal_nodes, 0, 0] = numba.uint64(x_min)
        tree_bounds[ip + num_internal_nodes, 1, 0] = numba.uint64(y_min)
        tree_bounds[ip + num_internal_nodes, 2, 0] = numba.uint64(z_min)
        tree_bounds[ip + num_internal_nodes, 0, 1] = numba.uint64(x_max)
        tree_bounds[ip + num_internal_nodes, 1, 1] = numba.uint64(y_max)
        tree_bounds[ip + num_internal_nodes, 2, 1] = numba.uint64(z_max)


@cuda.jit
def propagate_bounds_upwards(tree_bounds, tree_parents, num_leafs, num_internal_nodes):
    ip = cuda.grid(1)
    # Launch tread for each *leaf*
    if ip < num_leafs:
        child = ip + num_internal_nodes
        next_parent = tree_parents[child]

        while next_parent != -1:  # indicates hitting root
            x_min = tree_bounds[child, 0, 0]
            y_min = tree_bounds[child, 1, 0]
            z_min = tree_bounds[child, 2, 0]
            x_max = tree_bounds[child, 0, 1]
            y_max = tree_bounds[child, 1, 1]
            z_max = tree_bounds[child, 2, 1]
            cuda.atomic.min(tree_bounds, (next_parent, 0, 0), x_min)
            cuda.atomic.min(tree_bounds, (next_parent, 1, 0), y_min)
            cuda.atomic.min(tree_bounds, (next_parent, 2, 0), z_min)
            cuda.atomic.max(tree_bounds, (next_parent, 0, 1), x_max)
            cuda.atomic.max(tree_bounds, (next_parent, 1, 1), y_max)
            cuda.atomic.max(tree_bounds, (next_parent, 2, 1), z_max)

            # # This hopefully kills off one of the children threads
            # if child_min_key != tree_bounds[next_parent, 0]:
            #     break

            # Prepare next iteration of while loop
            child = next_parent
            next_parent = tree_parents[next_parent]


@cuda.jit(device=True, inline=True)
def distance(point, query_point):
    dist = math.sqrt((point[0] - query_point[0])**2
                     + (point[1] - query_point[1])**2
                     + (point[2] - query_point[2])**2)
    return dist


@cuda.jit(device=True, inline=True)
def is_point_in_box(point, box):
    for ii in range(3):
        if point[ii] < box[ii, 0] or point[ii] > box[ii, 1]:
            return False
    return True


@cuda.jit(device=True, inline=True)
def distance_to_box(point, box):
    """Squared distance from a point to an AABB (0 if inside)."""
    sq_dist = 0.0
    for i in range(3):
        if point[i] < box[i, 0]:
            d = box[i, 0] - point[i]
            sq_dist += d * d
        elif point[i] > box[i, 1]:
            d = point[i] - box[i, 1]
            sq_dist += d * d
    return math.sqrt(sq_dist)


@cuda.jit(device=True, inline=True)
def nearest_neighbor_device(points, tree_parents, tree_children, tree_bounds,
                            query_point, num_internal_nodes):
    """
    This is a nearest neighbor function which is fast and which
    works well on Voronoi cells.
    """

    # We traverse the nodes and leafs using a while loop and a queue.
    # Local memory on each tread (32 should be fine?)
    queue = cuda.local.array(128, numba.int64)
    # Initialize queue_index an start at node 0
    queue_index = 0
    queue[queue_index] = 0

    # Initialize min_dist, and min_index
    min_dist = (2**L - 1.0)
    min_index = -1

    while queue_index >= 0:

        node_id = queue[queue_index]
        # print(queue_index, node_id)#traverseA, traverseB, is_leafA, is_leafB)

        childA = tree_children[node_id, 0]
        childB = tree_children[node_id, 1]

        is_leafA = childA >= num_internal_nodes
        is_leafB = childB >= num_internal_nodes

        point_in_A = is_point_in_box(query_point, tree_bounds[childA])
        point_in_B = is_point_in_box(query_point, tree_bounds[childB])

        # Do explicit check if in a leaf
        if point_in_A and is_leafA:
            data_id = childA - num_internal_nodes
            dist = distance(points[data_id], query_point)

            if dist < min_dist:
                min_dist = dist
                min_index = data_id

        if point_in_B and is_leafB:
            data_id = childB - num_internal_nodes
            dist = distance(points[data_id], query_point)

            if dist < min_dist:
                min_dist = dist
                min_index = data_id

        # Whether to traverse
        traverseA = point_in_A and not is_leafA
        traverseB = point_in_B and not is_leafB

        if (not traverseA) and (not traverseB):
            queue_index -= 1
        else:
            if traverseA:
                queue[queue_index] = childA
            else:
                queue[queue_index] = childB
            if traverseA and traverseB:
                queue_index += 1
                queue[queue_index] = childB

    return min_dist, min_index


@cuda.jit(device=True, inline=True)
def nearest_neighbor_device_optimized(points, tree_parents, tree_children, tree_bounds,
                                      query_point, num_internal_nodes, min_dist_init):
    """
    This is a nearest neighbor function that works for general particle distributions,
    i.e., the points are *not* assumed to have overlapping bounding volumes that
    fill space with no holes.
    """

    # We traverse the nodes and leafs using a while loop and a queue.
    # Local memory on each tread (32 should be fine?)
    queue = cuda.local.array(128, numba.int64)
    # Initialize queue_index an start at node 0
    queue_index = 0
    queue[queue_index] = 0

    # Initialize min_dist, and min_index
    min_dist = min_dist_init
    min_index = -1

    while queue_index >= 0:

        node_id = queue[queue_index]

        childA = tree_children[node_id, 0]
        childB = tree_children[node_id, 1]

        is_leafA = childA >= num_internal_nodes
        is_leafB = childB >= num_internal_nodes

        # Do explicit check if in a leaf
        if is_leafA:
            data_id = childA - num_internal_nodes
            dist = distance(points[data_id], query_point)

            if dist < min_dist:
                min_dist = dist
                min_index = data_id

        if is_leafB:
            data_id = childB - num_internal_nodes
            dist = distance(points[data_id], query_point)

            if dist < min_dist:
                min_dist = dist
                min_index = data_id

        # Whether to traverse
        distA = distance_to_box(query_point, tree_bounds[childA])
        distB = distance_to_box(query_point, tree_bounds[childB])

        # Whether to traverse
        traverseA = (distA <= min_dist) and not is_leafA
        traverseB = (distB <= min_dist) and not is_leafB

        if (not traverseA) and (not traverseB):
            queue_index -= 1
        else:
            if traverseA:
                queue[queue_index] = childA
            else:
                queue[queue_index] = childB
            if traverseA and traverseB:
                queue_index += 1
                queue[queue_index] = childB

    return min_dist, min_index


@cuda.jit
def find_nearest_neighbors(points, tree_parents, tree_children, tree_bounds,
                           query_points, dists, ids):
    ip = cuda.grid(1)
    n_queries = query_points.shape[0]
    if ip < n_queries:
        num_internal_nodes = numba.int64(tree_children.shape[0])

        # Select a query_point from the list of queries
        query_point = query_points[ip]

        min_dist, min_index = nearest_neighbor_device(points, tree_parents, tree_children,
                                                      tree_bounds, query_point,
                                                      num_internal_nodes)

        dists[ip] = min_dist
        ids[ip] = min_index


def get_blocks(size, threadsperblock):
    return (size + (threadsperblock - 1)) // threadsperblock


class GpuBinaryTree:
    def __init__(self, positions, sizes, threadsperblock=32):
        """
        Python/Numba implementation of a BVH (boundary volume hierarchy) tree.
        """

        # TODO: Check if positions and sizes are
        # CuPy arrays. If not, automatically transfer.
        # If they are Paicos Quantities, store their
        # units so that the distance calculation can be returned with units.

        # Copy positions
        self._pos = cp.array(positions)

        # Convert self._position to units from [0, 1) * 2**L
        self.off_sets = cp.min(self._pos, axis=0)
        self._pos -= self.off_sets[None, :]

        max_pos = cp.max(self._pos)
        self.conversion_factor = (2**L - 1) / cp.float64(max_pos)
        self._pos *= self.conversion_factor

        # Some tree properties
        self.num_leafs = self._pos.shape[0]
        self.num_internal_nodes = self.num_leafs - 1
        self.num_leafs_and_nodes = 2 * self.num_leafs - 1

        # Find blocks per grid
        blocks_leafs = get_blocks(self.num_leafs, threadsperblock)
        blocks_nodes = get_blocks(self.num_internal_nodes, threadsperblock)

        # Calculate uint64 and Morton keys
        self._pos_uint = self._pos.astype(cp.uint64)
        self.morton_keys = cp.zeros(self.num_leafs, dtype=cp.uint64)
        get_morton_keys64[blocks_leafs, threadsperblock](
            self._pos_uint, self.morton_keys)

        # TODO: delete integer positions? Not really needed after this step

        # Store index for going back to original ids.
        self.sort_index = cp.argsort(self.morton_keys)

        # Do the sorting
        self.morton_keys = self.morton_keys[self.sort_index]
        self._pos = self._pos[self.sort_index, :]
        self._pos_uint = self._pos_uint[self.sort_index, :]

        del self._pos_uint

        # Allocate arrays
        self.children = -1 * cp.ones((self.num_internal_nodes, 2), dtype=int)
        self.parents = -1 * cp.ones(self.num_leafs_and_nodes, dtype=int)

        # This sets the parent and children properties
        generateHierarchy[blocks_nodes, threadsperblock](self.morton_keys, self.children,
                                                         self.parents)
        del self.morton_keys

    # Set the boundaries for the leafs
        self.bounds = cp.zeros(
            (self.num_leafs_and_nodes, 3, 2), dtype=cp.uint64)

        # Ensures that atomic min and max does not just set to starting values
        self.bounds[:, :, 0] = 2**(3 * L) - 1
        self.bounds[:, :, 1] = 0

        set_leaf_bounding_volumes[blocks_leafs, threadsperblock](self.bounds, self._pos,
                                                                 sizes[self.sort_index],
                                                                 self.conversion_factor)

        # Calculate the bounding volumes for internal nodes,
        # by propagating the information upwards in the tree
        propagate_bounds_upwards[blocks_leafs, threadsperblock](self.bounds, self.parents,
                                                                self.num_leafs, self.num_internal_nodes)

    def _to_tree_coordinates(self, pos):
        return (pos - self.off_sets[None, :]) * self.conversion_factor

    def _tree_node_ids_to_data_ids(self, ids):
        """
        Go from sorted ids to original ordering of
        *selected* data.
        """
        return cp.arange(self.sort_index.shape[0])[self.sort_index][ids]

    def nearest_neighbor(self, query_points, threadsperblock=32):
        # TODO: check whether query points are already on GPU

        if query_points.ndim == 1:
            n_queries = 1
        else:
            n_queries = query_points.shape[0]

        blocks = get_blocks(n_queries, threadsperblock)

        ids = cp.zeros(n_queries, dtype=cp.int64)
        dists = cp.zeros(n_queries, dtype=cp.float64)
        find_nearest_neighbors[blocks, threadsperblock](
            self._pos, self.parents, self.children, self.bounds,
            self._to_tree_coordinates(query_points), dists, ids)
        return dists / self.conversion_factor, self._tree_node_ids_to_data_ids(ids)

    def __del__(self):
        """
        Clean up like this? Apparently quite sporadic
        when  __del__ is called.
        """
        self.release_gpu_memory()

    def release_gpu_memory(self):
        if hasattr(self, '_pos'):
            del self._pos
        if hasattr(self, '_pos_uint'):
            del self._pos_uint
        if hasattr(self, 'morton_keys'):
            del self.morton_keys
        if hasattr(self, 'sort_index'):
            del self.sort_index
        if hasattr(self, 'children'):
            del self.children
        if hasattr(self, 'parents'):
            del self.parents
        if hasattr(self, 'bounds'):
            del self.bounds
        cp._default_memory_pool.free_all_blocks()


if __name__ == '__main__':
    import paicos as pa
    from scipy.spatial import KDTree
    import numpy as np
    pa.use_units(False)
    snap = pa.Snapshot(pa.data_dir + 'snap_247.hdf5')
    center = snap.Cat.Group['GroupPos'][0]
    widths = np.array([15000, 15000, 15000])

    index = pa.util.get_index_of_cubic_region_plus_thin_layer(
        snap['0_Coordinates'], center, widths,
        snap['0_Diameters'], snap.box)

    snap = snap.select(index, parttype=0)

    pos_cpu = snap['0_Coordinates']
    pos = cp.array(pos_cpu)
    sizes = cp.array(snap['0_Diameters'])
    bvh_tree = GpuBinaryTree(pos, 1.2 * sizes)

    bvh_dist, bvh_ids = bvh_tree.nearest_neighbor(pos)

    # Construct a scipy kd tree
    kd_tree = KDTree(pos_cpu)

    kd_dist, kd_ids = kd_tree.query(pos_cpu)

    np.testing.assert_array_equal(kd_ids, bvh_ids.get())

    # Random query positions
    pos2_cpu = np.random.rand(10000, 3) * widths[None, :]
    pos2_cpu += (center - widths / 2)[None, :]

    pos2 = cp.array(pos2_cpu)

    bvh_dist2, bvh_ids2 = bvh_tree.nearest_neighbor(pos2)
    kd_dist2, kd_ids2 = kd_tree.query(pos2_cpu)

    np.testing.assert_array_equal(kd_ids2, bvh_ids2.get())
