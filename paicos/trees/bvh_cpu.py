import numba
import math
import numpy as np
from .bvh_tree import leading_zeros_cython as count_leading_zeros
from .bvh_tree import generateHierarchy, propagate_bounds_upwards
from .bvh_tree import set_leaf_bounding_volumes
from .bvh_tree import get_morton_keys64
from .. import util
from numba import uint64

import ctypes
from numba.extending import get_cython_function_address

# Hardcoded for uint64 coordinates (don't change without also changing morton
# code functions)
L = 21

addr = get_cython_function_address("paicos.trees.bvh_tree", "leading_zeros_cython_for_numba")
functype = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_ulong)
leading_zeros_cython_for_numba = functype(addr)


@numba.njit(inline='always')
def count_leading_zeros_numba_inline(x: uint64) -> int:
    """
    """
    if x == 0:
        return 64
    n = 0
    if (x & 0xFFFFFFFF00000000) == 0:
        n += 32
        x <<= 32
    if (x & 0xFFFF000000000000) == 0:
        n += 16
        x <<= 16
    if (x & 0xFF00000000000000) == 0:
        n += 8
        x <<= 8
    if (x & 0xF000000000000000) == 0:
        n += 4
        x <<= 4
    if (x & 0xC000000000000000) == 0:
        n += 2
        x <<= 2
    if (x & 0x8000000000000000) == 0:
        n += 1
    return n


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
def get_morton_keys64_old(pos):
    morton_keys = np.empty(pos.shape[0], dtype=np.uint64)
    for ip in range(pos.shape[0]):
        morton_keys[ip] = encode64(pos[ip, 0], pos[ip, 1], pos[ip, 2])
    return morton_keys


# @numba.jit(nopython=True, inline='always')
def findSplit_old(sortedMortonCodes, first, last):
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
def generateHierarchy_old(sortedMortonCodes, tree_children, tree_parents):
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
        first, last = determineRange_old(sortedMortonCodes, n_codes, idx)

        # Determine where to split the range.
        split = findSplit_old(sortedMortonCodes, first, last)

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
def determineRange_old(sortedMortonCodes, n_codes, idx):
    """
    The determine range function needed by the generateHierarchy function.
    This code has has been adapted from the CornerStone CUDA/C++ code,
    see constructInternalNode in btree.hpp.
    """
    d = 1
    minPrefixLength = -1
    firstIndex = idx
    max_last = n_codes

    # print("python:", firstIndex)

    if (firstIndex > 0):
        p1 = sortedMortonCodes[firstIndex] ^ sortedMortonCodes[firstIndex + 1]
        m1 = sortedMortonCodes[firstIndex] ^ sortedMortonCodes[firstIndex - 1]

        # print("python:", p1, m1)

        # Count leading zeros
        lz_p1 = count_leading_zeros(p1)
        lz_m1 = count_leading_zeros(m1)

        # print("python:", lz_p1, lz_m1)

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
            if (0 <= secondIndex and secondIndex < max_last):
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


def set_leaf_bounding_volumes_old(tree_bounds, points, half_size, conversion_factor):
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


def propagate_bounds_upwards_old(tree_bounds, tree_parents, tree_children):
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


@numba.njit(inline='always')
def distance(point, query_point):
    dist = math.sqrt((point[0] - query_point[0])**2
                     + (point[1] - query_point[1])**2
                     + (point[2] - query_point[2])**2)
    return dist


@numba.njit(inline='always')
def is_point_in_box(point, box):
    for ii in range(3):
        if point[ii] < box[ii, 0] or point[ii] > box[ii, 1]:
            return False
    return True


@numba.njit(inline='always')
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


@numba.jit(nopython=True)
def box_intersects_sphere(box, center, radius):
    sq_dist = 0.0
    r2 = radius * radius
    for i in range(3):
        if center[i] < box[i, 0]:
            d = box[i, 0] - center[i]
            sq_dist += d * d
        elif center[i] > box[i, 1]:
            d = center[i] - box[i, 1]
            sq_dist += d * d
        if sq_dist > r2:
            return False
    return True


@numba.njit(inline='always')
def nearest_neighbor_cpu(points, tree_parents, tree_children, tree_bounds,
                         query_point, num_internal_nodes):
    queue = np.empty(128, dtype=np.int64)
    queue_index = 0
    queue[queue_index] = 0

    # Initialize min_dist, and min_index
    min_dist = (2**L - 1.0)
    min_index = -1

    while queue_index >= 0:
        node_id = queue[queue_index]

        childA = tree_children[node_id, 0]
        childB = tree_children[node_id, 1]

        is_leafA = childA >= num_internal_nodes
        is_leafB = childB >= num_internal_nodes

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

        distA = distance_to_box(query_point, tree_bounds[childA])
        distB = distance_to_box(query_point, tree_bounds[childB])

        # Whether to traverse
        traverseA = (distA <= min_dist) and not is_leafA
        traverseB = (distB <= min_dist) and not is_leafB

        if not traverseA and not traverseB:
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


@numba.njit(inline='always')
def nearest_neighbor_cpu_optimized(points, tree_parents, tree_children, tree_bounds,
                                   query_point, num_internal_nodes,
                                   min_dist_init):
    queue = np.empty(128, dtype=np.int64)
    queue_index = 0
    queue[queue_index] = 0

    # Start with optionally pre-supplied minimum distance (squared)
    min_dist2 = min_dist_init * min_dist_init
    min_index = -1

    while queue_index >= 0:
        node_id = queue[queue_index]

        childA = tree_children[node_id, 0]
        childB = tree_children[node_id, 1]

        is_leafA = childA >= num_internal_nodes
        is_leafB = childB >= num_internal_nodes

        # Handle leaf A
        if is_leafA:
            data_id = childA - num_internal_nodes
            dx = points[data_id, 0] - query_point[0]
            dy = points[data_id, 1] - query_point[1]
            dz = points[data_id, 2] - query_point[2]
            dist2 = dx * dx + dy * dy + dz * dz
            if dist2 < min_dist2:
                min_dist2 = dist2
                min_index = data_id

        # Handle leaf B
        if is_leafB:
            data_id = childB - num_internal_nodes
            dx = points[data_id, 0] - query_point[0]
            dy = points[data_id, 1] - query_point[1]
            dz = points[data_id, 2] - query_point[2]
            dist2 = dx * dx + dy * dy + dz * dz
            if dist2 < min_dist2:
                min_dist2 = dist2
                min_index = data_id

        # Compute distances to AABBs (squared)
        distA2 = 0.0
        for i in range(3):
            if query_point[i] < tree_bounds[childA, i, 0]:
                d = tree_bounds[childA, i, 0] - query_point[i]
                distA2 += d * d
            elif query_point[i] > tree_bounds[childA, i, 1]:
                d = query_point[i] - tree_bounds[childA, i, 1]
                distA2 += d * d

        distB2 = 0.0
        for i in range(3):
            if query_point[i] < tree_bounds[childB, i, 0]:
                d = tree_bounds[childB, i, 0] - query_point[i]
                distB2 += d * d
            elif query_point[i] > tree_bounds[childB, i, 1]:
                d = query_point[i] - tree_bounds[childB, i, 1]
                distB2 += d * d

        # Decide whether to traverse children
        traverseA = (distA2 <= min_dist2) and not is_leafA
        traverseB = (distB2 <= min_dist2) and not is_leafB

        if not traverseA and not traverseB:
            queue_index -= 1
        else:
            # Choose traversal order based on proximity
            if traverseA and traverseB:
                if distA2 < distB2:
                    queue[queue_index] = childA
                    queue_index += 1
                    queue[queue_index] = childB
                else:
                    queue[queue_index] = childB
                    queue_index += 1
                    queue[queue_index] = childA
            elif traverseA:
                queue[queue_index] = childA
            elif traverseB:
                queue[queue_index] = childB

    return math.sqrt(min_dist2), min_index


@numba.njit(inline='always')
def nearest_neighbor_cpu_voronoi(points, tree_parents, tree_children, tree_bounds,
                                 query_point, num_internal_nodes):
    queue = np.empty(128, dtype=np.int64)
    queue_index = 0
    queue[queue_index] = 0

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

        if not traverseA and not traverseB:
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


@numba.jit(nopython=True)
def find_points_in_range(points, tree_children, tree_bounds,
                         query_points, radius, max_neighbors):
    n_queries = query_points.shape[0]
    num_internal_nodes = tree_children.shape[0]

    all_neighbors = np.full((n_queries, max_neighbors), -1, dtype=np.int64)
    neighbor_counts = np.zeros(n_queries, dtype=np.int64)

    for ip in range(n_queries):
        query_point = query_points[ip]

        queue = np.zeros(256, dtype=np.int64)
        queue_index = 0
        queue[queue_index] = 0  # Start from root

        while queue_index >= 0:
            node_id = queue[queue_index]
            childA = tree_children[node_id, 0]
            childB = tree_children[node_id, 1]

            is_leafA = childA >= num_internal_nodes
            is_leafB = childB >= num_internal_nodes

            intersectsA = box_intersects_sphere(tree_bounds[childA], query_point, radius)
            intersectsB = box_intersects_sphere(tree_bounds[childB], query_point, radius)

            if intersectsA and is_leafA:
                data_id = childA - num_internal_nodes
                dist = distance(query_point, points[data_id])
                if dist <= radius:
                    count = neighbor_counts[ip]
                    if count < max_neighbors:
                        all_neighbors[ip, count] = data_id
                        neighbor_counts[ip] += 1

            if intersectsB and is_leafB:
                data_id = childB - num_internal_nodes
                dist = distance(query_point, points[data_id])
                if dist <= radius:
                    count = neighbor_counts[ip]
                    if count < max_neighbors:
                        all_neighbors[ip, count] = data_id
                        neighbor_counts[ip] += 1

            traverseA = intersectsA and not is_leafA
            traverseB = intersectsB and not is_leafB

            if not traverseA and not traverseB:
                queue_index -= 1
            else:
                if traverseA:
                    queue[queue_index] = childA
                else:
                    queue[queue_index] = childB
                if traverseA and traverseB:
                    queue_index += 1
                    queue[queue_index] = childB

    return all_neighbors, neighbor_counts


@numba.jit(nopython=True)
def find_nearest_neighbors(points, tree_parents, tree_children, tree_bounds,
                           query_points, dists, ids, start_id=0, ignore_self=False):
    n_queries = query_points.shape[0]
    num_internal_nodes = numba.int64(tree_children.shape[0])

    for ip in range(n_queries):

        # Select a query_point from the list of queries
        query_point = query_points[ip]

        this_query_start_id = start_id

        # We traverse the nodes and leafs using a while loop and a queue.

        # Local memory on each tread (32 should be fine?)
        queue = np.zeros(256, dtype=np.int64)
        # Initialize queue_index an start at node 0
        queue_index = 0
        queue[queue_index] = this_query_start_id

        # Initialize min_dist, and min_index
        min_dist = (2**L - 1.0)
        min_index = -1

        # print('\n\nstarting\n\n')

        while queue_index >= 0:

            node_id = queue[queue_index]

            childA = tree_children[node_id, 0]
            childB = tree_children[node_id, 1]

            is_leafA = childA >= num_internal_nodes
            is_leafB = childB >= num_internal_nodes

            if is_leafA:
                data_id = childA - num_internal_nodes
                dist = distance(query_point, points[data_id])

                if dist < min_dist:
                    if (dist > 0.0) or (not ignore_self):
                        min_dist = dist
                        min_index = data_id

            if is_leafB:
                data_id = childB - num_internal_nodes
                dist = distance(query_point, points[data_id])

                if dist < min_dist:
                    if (dist > 0.0) or (not ignore_self):
                        min_dist = dist
                        min_index = data_id

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

        # Outside main while loop
        dists[ip] = min_dist
        ids[ip] = min_index  # + num_internal_nodes + 1
    return dists, ids


class BinaryTree:
    def __init__(self, positions, sizes, verbose=False):
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
        if verbose:
            print('Tree: Generating morton keys')
        self.morton_keys = get_morton_keys64(self._pos_uint)
        # print(self.morton_keys)

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

        if verbose:
            print("Tree: Generate parent/children hierarchy")
        # This sets the parent and children properties
        generateHierarchy(self.morton_keys, self.children, self.parents)

        # Set the boundaries for the leafs
        self.bounds = np.zeros((self.num_leafs_and_nodes, 3, 2), dtype=np.uint64)

        if verbose:
            print("Tree: Set leaf bounding volumes")
        set_leaf_bounding_volumes(self.bounds, self._pos,
                                  sizes[self.sort_index],
                                  self.conversion_factor, L)

        # Calculate the bounding volumes for internal nodes,
        # by propagating the information upwards in the tree
        if verbose:
            print("Tree: Propagating bounds")
        propagate_bounds_upwards(self.bounds, self.parents, self.children)

        if verbose:
            print("Tree: Construction [DONE]")

    def _to_tree_coordinates(self, pos):
        return (pos - self.off_sets[None, :]) * self.conversion_factor

    def _tree_node_ids_to_data_ids(self, ids):
        """
        Go from sorted ids to original ordering of
        *selected* data.
        """
        return np.arange(self.sort_index.shape[0])[self.sort_index][ids]

    def _data_ids_to_tree_node_ids(self, original_ids):
        """
        Go from original data indices of selected data
        to sorted indices (tree node ids).
        """
        inverse_index = np.empty_like(self.sort_index)
        inverse_index[self.sort_index] = np.arange(self.sort_index.shape[0])
        return inverse_index[original_ids]

    @util.conditional_timer
    def nearest_neighbor(self, query_points, start_id=0, ignore_self=False, timing=False):
        if query_points.ndim == 1:
            n_queries = 1
        else:
            n_queries = query_points.shape[0]

        ids = np.zeros(n_queries, dtype=np.int64)
        dists = np.zeros(n_queries, dtype=np.float64)
        find_nearest_neighbors(self._pos, self.parents, self.children, self.bounds,
                               self._to_tree_coordinates(
                                   query_points), dists, ids,
                               start_id=start_id, ignore_self=ignore_self)
        return dists / self.conversion_factor, self._tree_node_ids_to_data_ids(ids)

    def range_search(self, query_points, radius, max_neighbors=256):
        if query_points.ndim == 1:
            query_points = query_points[None, :]

        raw_neighbors, neighbor_counts = find_points_in_range(
            self._pos, self.children, self.bounds,
            self._to_tree_coordinates(query_points), radius * self.conversion_factor, max_neighbors
        )

        neighbor_lists = []
        for i in range(len(query_points)):
            n_valid = neighbor_counts[i]
            ids = raw_neighbors[i, :n_valid]  # Only keep valid entries
            neighbor_lists.append(self._tree_node_ids_to_data_ids(ids))

        if neighbor_counts.max() >= max_neighbors:
            import warnings
            warning_msg = (f"Range search has max_neighbors={max_neighbors}"
                           + f" but the largest number of neighbors found was {neighbor_counts.max()}.")
            warnings.warn(warning_msg)

        return neighbor_lists, neighbor_counts
