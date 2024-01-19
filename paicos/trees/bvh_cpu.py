import numba
import math
import numpy as np
from .bvh_tree import leading_zeros_cython as count_leading_zeros

# Hardcoded for uint64 coordinates (don't change without also changing morton
# code functions)
L = 21


def leading_zeros_python(key64bit):
    """
    1 us per calculation, so super slow
    compared to the 3 ns per calculation that one gets
    with Cython.
    """
    return format(key64bit, '064b').find('1')


@numba.jit(nopython=True)
def part1by2_64(n):
    n &= 0x1fffff
    n = (n | (n << 32)) & 0x1f00000000ffff
    n = (n | (n << 16)) & 0x1f0000ff0000ff
    n = (n | (n << 8)) & 0x100f00f00f00f00f
    n = (n | (n << 4)) & 0x10c30c30c30c30c3
    n = (n | (n << 2)) & 0x1249249249249249
    return n


@numba.jit(nopython=True)
def unpart1by2_64(n):
    n = numba.uint64(n)
    n &= 0x1249249249249249
    n = (n ^ (n >> 2)) & 0x10c30c30c30c30c3
    n = (n ^ (n >> 4)) & 0x100f00f00f00f00f
    n = (n ^ (n >> 8)) & 0x1f0000ff0000ff
    n = (n ^ (n >> 16)) & 0x1f00000000ffff
    n = (n ^ (n >> 32)) & 0x1fffff
    return n


@numba.jit(nopython=True)
def encode64(x, y, z):
    xx = part1by2_64(x)
    yy = part1by2_64(y)
    zz = part1by2_64(z)
    return xx * 4 + yy * 2 + zz


@numba.jit(nopython=True)
def decode64(key):
    x = unpart1by2_64(key >> 2)
    y = unpart1by2_64(key >> 1)
    z = unpart1by2_64(key)
    return numba.float64(x), numba.float64(y), numba.float64(z)


@numba.jit(nopython=True)
def get_morton_keys64(pos):
    morton_keys = np.empty(pos.shape[0], dtype=np.uint64)
    for ip in range(pos.shape[0]):
        morton_keys[ip] = encode64(pos[ip, 0], pos[ip, 1], pos[ip, 2])
    return morton_keys


# @numba.jit(nopython=True, inline='always')
def findSplit(sortedMortonCodes, first, last):
    # Identical Morton codes => split the range in the middle.

    firstCode = sortedMortonCodes[first]
    lastCode = sortedMortonCodes[last]

    if (firstCode == lastCode):
        return (first + last) >> 1

    # Calculate the number of highest bits that are the same
    # for all objects, using the count-leading-zeros intrinsic.

    commonPrefix = count_leading_zeros(firstCode ^ lastCode)

    # Use binary search to find where the next bit differs.
    # Specifically, we are looking for the highest object that
    # shares more than commonPrefix bits with the first one.

    split = first  # initial guess
    step = last - first

    while (step > 1):
        step = (step + 1) >> 1  # exponential decrease
        newSplit = split + step  # proposed new position

        if (newSplit < last):
            splitCode = sortedMortonCodes[newSplit]
            splitPrefix = count_leading_zeros(firstCode ^ splitCode)
            if (splitPrefix > commonPrefix):
                split = newSplit  # accept proposal

    return split


# @numba.jit(nopython=True)
def generateHierarchy(sortedMortonCodes, tree_children, tree_parents):
    """
    Generate tree using the sorted Morton code.
    This is done in parallel, see this blogpost
    https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    and the corresponding paper:

    """
    n_codes = sortedMortonCodes.shape[0]
    num_internal_nodes = n_codes - 1

    # Launch thread for each internal node
    for idx in range(num_internal_nodes):

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


# @numba.jit(nopython=True, inline='always')
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
        p1 = sortedMortonCodes[firstIndex] ^ sortedMortonCodes[firstIndex + 1]
        m1 = sortedMortonCodes[firstIndex] ^ sortedMortonCodes[firstIndex - 1]

        # Count leading zeros
        lz_p1 = count_leading_zeros(p1)
        lz_m1 = count_leading_zeros(m1)

        if lz_p1 > lz_m1:
            d = 1
            minPrefixLength = lz_m1
        else:
            d = -1
            minPrefixLength = lz_p1

    searchRange = 2
    secondIndex = firstIndex + searchRange * d

    while (0 <= secondIndex and secondIndex < max_last):
        lz_first_second = count_leading_zeros(
            sortedMortonCodes[firstIndex] ^ sortedMortonCodes[secondIndex])
        if lz_first_second > minPrefixLength:
            searchRange *= 2
            secondIndex = firstIndex + searchRange * d
            if secondIndex < max_last:
                lz_first_second = count_leading_zeros(
                    sortedMortonCodes[firstIndex] ^ sortedMortonCodes[secondIndex])
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
            lz_f_Jdx = count_leading_zeros(
                sortedMortonCodes[firstIndex] ^ sortedMortonCodes[newJdx])
            if lz_f_Jdx > minPrefixLength:
                secondIndex = newJdx

    first = min(secondIndex, firstIndex)
    last = max(secondIndex, firstIndex)

    return first, last


def find_bounding_volume_as_sfc_keys(center, hsml):
    min_val_x = numba.uint64(max(center[0] - hsml, 0))
    min_val_y = numba.uint64(max(center[1] - hsml, 0))
    min_val_z = numba.uint64(max(center[2] - hsml, 0))
    max_val_x = numba.uint64(min(math.ceil(center[0] + hsml), 2**L - 1))
    max_val_y = numba.uint64(min(math.ceil(center[1] + hsml), 2**L - 1))
    max_val_z = numba.uint64(min(math.ceil(center[2] + hsml), 2**L - 1))
    sfc_key_min = encode64(min_val_x, min_val_y, min_val_z)
    sfc_key_max = encode64(max_val_x, max_val_y, max_val_z)
    return sfc_key_min, sfc_key_max


def set_leaf_bounding_volumes(tree_bounds, points, half_size, conversion_factor):
    num_internal_nodes = half_size.shape[0] - 1
    num_leafs = half_size.shape[0]
    f = conversion_factor
    to_int = numba.uint64
    for ip in range(num_leafs):
        center = points[ip, :]
        x_min = to_int(max((center[0] - f * half_size[ip]), 0))
        y_min = to_int(max((center[1] - f * half_size[ip]), 0))
        z_min = to_int(max((center[2] - f * half_size[ip]), 0))
        x_max = to_int(
            min(math.ceil((center[0] + f * half_size[ip])), 2**L - 1))
        y_max = to_int(
            min(math.ceil((center[1] + f * half_size[ip])), 2**L - 1))
        z_max = to_int(
            min(math.ceil((center[2] + f * half_size[ip])), 2**L - 1))
        tree_bounds[ip + num_internal_nodes, 0, 0] = x_min
        tree_bounds[ip + num_internal_nodes, 1, 0] = y_min
        tree_bounds[ip + num_internal_nodes, 2, 0] = z_min
        tree_bounds[ip + num_internal_nodes, 0, 1] = x_max
        tree_bounds[ip + num_internal_nodes, 1, 1] = y_max
        tree_bounds[ip + num_internal_nodes, 2, 1] = z_max


def propagate_bounds_upwards(tree_bounds, tree_parents, tree_children):
    num_internal_nodes = tree_children.shape[0]
    num_leafs = num_internal_nodes + 1

    next_parents = []
    for ip in range(num_leafs):
        next_parents.append(tree_parents[ip + num_internal_nodes])
    next_parents = np.unique(next_parents)

    box = np.zeros((3, 2), dtype=np.uint64)
    while len(next_parents) > 0:
        current_parents = np.array(next_parents)
        next_parents = []
        for ii in range(current_parents.shape[0]):
            node_id = current_parents[ii]
            next_parent = tree_parents[node_id]
            if next_parent != -1:
                next_parents.append(tree_parents[node_id])
            childA = tree_children[node_id, 0]
            childB = tree_children[node_id, 1]
            boundsA = tree_bounds[childA, :, :]
            boundsB = tree_bounds[childB, :, :]
            box[0, 0] = np.min([boundsB[0, 0], boundsA[0, 0]])
            box[1, 0] = np.min([boundsB[1, 0], boundsA[1, 0]])
            box[2, 0] = np.min([boundsB[2, 0], boundsA[2, 0]])
            box[0, 1] = np.max([boundsB[0, 1], boundsA[0, 1]])
            box[1, 1] = np.max([boundsB[1, 1], boundsA[1, 1]])
            box[2, 1] = np.max([boundsB[2, 1], boundsA[2, 1]])

            tree_bounds[node_id, :, :] = box[:, :]
        next_parents = np.unique(next_parents)
        # if len(next_parents) < 10:
        #     print(node_id, len(next_parents), next_parents)


@numba.jit(nopython=True)
def distance(xf_q, yf_q, zf_q, xf, yf, zf):
    dist = math.sqrt((xf - xf_q)**2
                     + (yf - yf_q)**2
                     + (zf - zf_q)**2)
    return dist


# @numba.jit(nopython=True)
# def is_within_bounds(xf_q, yf_q, zf_q, tree_bound):
#     x_min, y_min, z_min = tree_bound[:, ]decode64(tree_bound[0])
#     x_max, y_max, z_max = decode64(tree_bound[1])
#     if x_min > xf_q or x_max < xf_q:
#         return False
#     if y_min > yf_q or y_max < yf_q:
#         return False
#     if z_min > zf_q or z_max < zf_q:
#         return False
#     return True

@numba.jit(nopython=True)
def is_point_in_box(point, box):
    for ii in range(3):
        if point[ii] < box[ii, 0] or point[ii] > box[ii, 1]:
            return False
    return True


@numba.jit(nopython=True)
def find_nearest_neighbors(points, tree_parents, tree_children, tree_bounds,
                           query_points, dists, ids, start_id=0):
    n_queries = query_points.shape[0]
    for ip in range(n_queries):
        num_internal_nodes = numba.int64(tree_children.shape[0])

        # Select a query_point from the list of queries
        query_point = query_points[ip]

        if start_id != 0:
            this_query_start_id = int(start_id)
            # print(this_query_start_id, start_id)
            is_leaf = start_id >= num_internal_nodes
            in_box = is_point_in_box(
                query_point, tree_bounds[this_query_start_id])
            while not in_box or is_leaf:
                # print(this_query_start_id, start_id)
                this_query_start_id = tree_parents[this_query_start_id]
                # print('dfdf')
                in_box = is_point_in_box(
                    query_point, tree_bounds[this_query_start_id])
                # print('22222')
                is_leaf = this_query_start_id >= num_internal_nodes
                # print('33333', is_leaf, in_box)
                # print(this_query_start_id, start_id)
                if this_query_start_id == -1:
                    # print('went all the way up...')
                    this_query_start_id = 0
                    break
        else:
            this_query_start_id = 0

        # print('did we go here?')
        # print('this_query_start_id', this_query_start_id, start_id)
        # We traverse the nodes and leafs using a while loop and a queue.

        # Local memory on each tread (32 should be fine?)
        queue = np.zeros(256, dtype=np.int64)
        # Initialize queue_index an start at node 0
        queue_index = 0
        queue[queue_index] = this_query_start_id

        # Initialize min_dist, and min_index
        min_dist = (2**L - 1.0)
        min_index = -1

        while queue_index >= 0:

            node_id = queue[queue_index]

            childA = tree_children[node_id, 0]
            childB = tree_children[node_id, 1]

            is_leafA = childA >= num_internal_nodes
            is_leafB = childB >= num_internal_nodes

            point_in_A = is_point_in_box(query_point, tree_bounds[childA])
            point_in_B = is_point_in_box(query_point, tree_bounds[childB])

            # Do explicit check if in a leaf
            # print(f'node_id: {node_id}\t childA: {childA}\t childB: {childB}\t')
            # if node_id == 6:
            #     print('node_id, point_in_A, point_in_B, is_leaf,
            # is_leafB', node_id, point_in_A, point_in_B, is_leafA, is_leafB)
            #     print(query_point)
            #     print(tree_bounds[childA])
            #     print(tree_bounds[childB])
            # raise RuntimeError("d")
            if point_in_A and is_leafA:
                data_id = childA - num_internal_nodes
                dist = distance(query_point[0],
                                query_point[1],
                                query_point[2],
                                points[data_id, 0],
                                points[data_id, 1],
                                points[data_id, 2])

                # if node_id == 6:
                #     print(f'dist: {dist}, min_dist: {min_dist}')
                #     print(dist, min_dist, data_id)
                #     print(points[data_id], query_point)
                # raise RuntimeError("d")
                if dist < min_dist:
                    min_dist = dist
                    min_index = data_id

            if point_in_B and is_leafB:
                data_id = childB - num_internal_nodes
                dist = distance(query_point[0],
                                query_point[1],
                                query_point[2],
                                points[data_id, 0],
                                points[data_id, 1],
                                points[data_id, 2])

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

        # Outside main while loop
        dists[ip] = min_dist
        ids[ip] = min_index  # + num_internal_nodes + 1
    return dists, ids


class BinaryTree:
    def __init__(self, positions, sizes):
        """
        Python/Numba implementation of a BVH (boundary volume hierarchy) tree.
        """

        # Copy array
        self._pos = np.array(positions)

        # Convert self._position to units from [0, 1) * 2**L
        self.off_sets = np.min(self._pos, axis=0)
        self._pos -= self.off_sets[None, :]

        max_pos = np.max(self._pos)
        self.conversion_factor = (2**L - 1) / max_pos
        self._pos *= self.conversion_factor

        # if use_units take the value...
        # self.pos = self.pos.value
        # self.

        # Calculate uint64 and Morton keys
        self._pos_uint = self._pos.astype(np.uint64)
        self.morton_keys = get_morton_keys64(self._pos_uint)

        # Store index for going back to original ids.
        self.sort_index = np.argsort(self.morton_keys)

        # Do the sorting
        self.morton_keys = self.morton_keys[self.sort_index]
        self._pos = self._pos[self.sort_index, :]
        self._pos_uint = self._pos_uint[self.sort_index, :]

        # Some tree properties
        self.num_leafs = self.morton_keys.shape[0]
        self.num_internal_nodes = self.num_leafs - 1
        self.num_leafs_and_nodes = 2 * self.num_leafs - 1

        # Allocate arrays
        self.children = -1 * np.ones((self.num_internal_nodes, 2), dtype=int)
        self.parents = -1 * np.ones(self.num_leafs_and_nodes, dtype=int)

        # This sets the parent and children properties
        generateHierarchy(self.morton_keys, self.children, self.parents)

        # Set the boundaries for the leafs
        self.bounds = np.zeros((self.num_leafs_and_nodes, 3, 2), dtype=np.uint64)

        set_leaf_bounding_volumes(self.bounds, self._pos,
                                  sizes[self.sort_index],
                                  self.conversion_factor)

        # Calculate the bounding volumes for internal nodes,
        # by propagating the information upwards in the tree
        propagate_bounds_upwards(self.bounds, self.parents, self.children)

    def _to_tree_coordinates(self, pos):
        return (pos - self.off_sets[None, :]) * self.conversion_factor

    def _tree_node_ids_to_data_ids(self, ids):
        """
        Go from sorted ids to original ordering of
        *selected* data.
        """
        return np.arange(self.sort_index.shape[0])[self.sort_index][ids]

    def nearest_neighbor(self, query_points, start_id=0):
        if query_points.ndim == 1:
            n_queries = 1
        else:
            n_queries = query_points.shape[0]

        ids = np.zeros(n_queries, dtype=np.int64)
        dists = np.zeros(n_queries, dtype=np.float64)
        find_nearest_neighbors(self._pos, self.parents, self.children, self.bounds,
                               self._to_tree_coordinates(
                                   query_points), dists, ids,
                               start_id=start_id)
        return dists / self.conversion_factor, self._tree_node_ids_to_data_ids(ids)
