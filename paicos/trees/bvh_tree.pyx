import cython
import numpy as np
cimport numpy as np
np.import_array()
from cython.parallel import prange
from libc.math cimport fmin, fmax
from libc.math cimport ceil
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memset

ctypedef np.int32_t int32_t
ctypedef np.int64_t int64_t


ctypedef unsigned long ulong
ctypedef double real_t

cdef extern int __builtin_clzl(unsigned long x) nogil

cdef int L = 21

def leading_zeros_cython(unsigned long x):
    return __builtin_clzl(x)

cdef api long leading_zeros_cython_for_numba(unsigned long x):
    return <long>(__builtin_clzl(x))

def get_len_of_common_prefix_cython(ulong key1, ulong key2):
    return __builtin_clzl(key1 ^ key2)

def get_leading_zeros(unsigned long[:] morton_keys):
    cdef int ii, N
    N = morton_keys.shape[0] - 1
    cdef int[:] my_leading_zeros = np.zeros(N, dtype=np.int32)
    for ii in range(N):
        key = morton_keys[ii] ^ morton_keys[ii+1]
        my_leading_zeros[ii] = __builtin_clzl(key)
    tmp = np.zeros(N, dtype=np.int32)
    tmp[:] = my_leading_zeros[:]
    return tmp

def set_leaf_bounding_volumes(ulong[:, :, :] tree_bounds,
                               real_t[:, :] points,
                               real_t[:] half_size,
                               real_t conversion_factor,
                               ulong L):
    # cdef ulong L = 21
    cdef Py_ssize_t ip
    cdef Py_ssize_t num_leafs = half_size.shape[0]
    cdef Py_ssize_t num_internal_nodes = num_leafs - 1

    cdef real_t f = conversion_factor
    cdef real_t cx, cy, cz, hs
    cdef ulong max_index = (<ulong>1 << L) - 1
    cdef ulong x_min, y_min, z_min, x_max, y_max, z_max

    with nogil:
        for ip in prange(num_leafs, schedule='static'):
            hs = f * half_size[ip]
            cx = points[ip, 0]
            cy = points[ip, 1]
            cz = points[ip, 2]

            x_min = <ulong> max(cx - hs, 0.0)
            y_min = <ulong> max(cy - hs, 0.0)
            z_min = <ulong> max(cz - hs, 0.0)

            x_max = <ulong> min(ceil(cx + hs), max_index)
            y_max = <ulong> min(ceil(cy + hs), max_index)
            z_max = <ulong> min(ceil(cz + hs), max_index)

            tree_bounds[ip + num_internal_nodes, 0, 0] = x_min
            tree_bounds[ip + num_internal_nodes, 1, 0] = y_min
            tree_bounds[ip + num_internal_nodes, 2, 0] = z_min
            tree_bounds[ip + num_internal_nodes, 0, 1] = x_max
            tree_bounds[ip + num_internal_nodes, 1, 1] = y_max
            tree_bounds[ip + num_internal_nodes, 2, 1] = z_max



def propagate_bounds_upwards(ulong[:, :, :] tree_bounds,
                             int64_t[:] tree_parents,
                             int64_t[:, :] tree_children):
    cdef int num_internal_nodes = tree_children.shape[0]
    cdef int num_leafs = num_internal_nodes + 1
    cdef int num_total_nodes = num_internal_nodes + num_leafs

    cdef int64_t *queue = <int64_t *> malloc(num_total_nodes * sizeof(int64_t))
    cdef int64_t *next_queue = <int64_t *> malloc(num_total_nodes * sizeof(int64_t))
    cdef int *seen = <int *> malloc(num_total_nodes * sizeof(int))
    cdef int *child_counter = <int *> malloc(num_total_nodes * sizeof(int))
    memset(seen, 0, num_total_nodes * sizeof(int))
    memset(child_counter, 0, num_total_nodes * sizeof(int))

    cdef int i, node_id, parent_id
    cdef int queue_len = 0
    cdef int next_len = 0

    # Queue initialization: find unique parents of leaves
    for i in range(num_leafs):
        node_id = num_internal_nodes + i
        parent_id = tree_parents[node_id]
        if parent_id >= 0:
            child_counter[parent_id] += 1

    for i in range(num_internal_nodes):
        if child_counter[i] == 2:
            seen[i] = 1
            queue[queue_len] = i
            queue_len += 1

    cdef ulong[:, :, :] bounds = tree_bounds
    cdef int64_t[:] parents = tree_parents
    cdef int64_t[:, :] children = tree_children
    cdef int childA, childB, dim
    cdef ulong a_min, b_min, a_max, b_max

    while queue_len > 0:
        next_len = 0
        for i in range(queue_len):
            node_id = queue[i]
            parent_id = parents[node_id]
            if parent_id >= 0:
                child_counter[parent_id] += 1
                if child_counter[parent_id] == 2 and seen[parent_id] == 0:
                    seen[parent_id] = 1
                    next_queue[next_len] = parent_id
                    next_len += 1

            childA = children[node_id, 0]
            childB = children[node_id, 1]

            for dim in range(3):
                a_min = bounds[childA, dim, 0]
                b_min = bounds[childB, dim, 0]
                a_max = bounds[childA, dim, 1]
                b_max = bounds[childB, dim, 1]
                bounds[node_id, dim, 0] = min(a_min, b_min)
                bounds[node_id, dim, 1] = max(a_max, b_max)

        queue_len = next_len
        tmp = queue
        queue = next_queue
        next_queue = tmp

    free(queue)
    free(next_queue)
    free(seen)
    free(child_counter)

cdef inline int count_leading_zeros(unsigned long x) noexcept nogil:
    return __builtin_clzl(x)

cdef inline ulong part1by2_64(ulong n) noexcept nogil:
    n &= <ulong>0x1fffff
    n = (n | (n << 32)) & <ulong>0x1f00000000ffff
    n = (n | (n << 16)) & <ulong>0x1f0000ff0000ff
    n = (n | (n << 8)) & <ulong>0x100f00f00f00f00f
    n = (n | (n << 4)) & <ulong>0x10c30c30c30c30c3
    n = (n | (n << 2)) & <ulong>0x1249249249249249
    return n

cdef inline ulong unpart1by2_64(ulong n) noexcept nogil:
    n &= <ulong>0x1249249249249249
    n = (n ^ (n >> 2)) & <ulong>0x10c30c30c30c30c3
    n = (n ^ (n >> 4)) & <ulong>0x100f00f00f00f00f
    n = (n ^ (n >> 8)) & <ulong>0x1f0000ff0000ff
    n = (n ^ (n >> 16)) & <ulong>0x1f00000000ffff
    n = (n ^ (n >> 32)) & <ulong>0x1fffff
    return n

cdef inline ulong encode64(ulong x, ulong y, ulong z) noexcept nogil:
    return 4 * part1by2_64(x) + 2 * part1by2_64(y) + part1by2_64(z)

def encode64_cython(ulong x, ulong y, ulong z):
    return 4 * part1by2_64(x) + 2 * part1by2_64(y) + part1by2_64(z)

cdef inline void decode64(ulong key, ulong[:] out) noexcept nogil:
    out[0] = unpart1by2_64(key >> 2)
    out[1] = unpart1by2_64(key >> 1)
    out[2] = unpart1by2_64(key)

def get_morton_keys64(ulong[:, :] pos):
    cdef Py_ssize_t N = pos.shape[0]
    cdef ulong[:] morton_keys = np.empty(N, dtype=np.uint64)
    cdef Py_ssize_t i
    for i in range(N):
        morton_keys[i] = encode64(<ulong>pos[i,0], <ulong>pos[i,1], <ulong>pos[i,2])

    tmp = np.empty(N, dtype=np.uint64)
    tmp[:] = morton_keys[:]
    return tmp

cdef inline int delta(ulong[:] sortedMortonCodes, int i, int j) noexcept nogil:
    cdef ulong codeA = sortedMortonCodes[i]
    cdef ulong codeB = sortedMortonCodes[j]

    if codeA == codeB:
        # Fallback: compare indices instead of codes
        return 64 + count_leading_zeros(<ulong>(i ^ j))
    else:
        return count_leading_zeros(codeA ^ codeB)

cdef inline int findSplit(ulong[:] sortedMortonCodes, int first, int last) noexcept nogil:
    cdef int commonPrefix = delta(sortedMortonCodes, first, last)
    cdef int split = first
    cdef int step = last - first
    cdef int newSplit, splitPrefix

    while step > 1:
        step = (step + 1) >> 1
        newSplit = split + step
        if newSplit < last:
            splitPrefix = delta(sortedMortonCodes, first, newSplit)
            if splitPrefix > commonPrefix:
                split = newSplit

    return split

cdef inline void determineRange(ulong[:] sortedMortonCodes, int n_codes, int idx, int[:] first_last) noexcept nogil:
    cdef int d = 1, minPrefixLength = -1
    cdef int firstIndex = idx, max_last = n_codes
    cdef int lz_p1, lz_m1, lz_first_second, searchRange, secondIndex, newJdx
    cdef int first, last

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
            if (0 <= secondIndex and secondIndex < max_last):
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
    first_last[0] = first
    first_last[1] = last

def generateHierarchy(ulong[:] sortedMortonCodes, long[:, :] tree_children, long[:] tree_parents):
    cdef int n_codes = sortedMortonCodes.shape[0]
    cdef int num_internal_nodes = n_codes - 1
    cdef int idx, first, last, split
    cdef int childA, childB

    cdef int[:] first_last = np.empty(2, dtype=np.int32)
    # cdef int[2] first_last

    for idx in range(num_internal_nodes):
        determineRange(sortedMortonCodes, n_codes, idx, first_last)
        first = first_last[0]
        last = first_last[1]
        split = findSplit(sortedMortonCodes, first, last)

        # Select childA
        if split == first:
            # Is a leaf
            childA = split + num_internal_nodes
        else:
            # Is in an internal node
            childA = split

        # Select childB
        if split + 1 == last:
            childB = split + 1 + num_internal_nodes
        else:
            childB = split + 1

        # Set children and parent
        tree_children[idx, 0] = childA
        tree_children[idx, 1] = childB
        tree_parents[childA] = idx
        tree_parents[childB] = idx
