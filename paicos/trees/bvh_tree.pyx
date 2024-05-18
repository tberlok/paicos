import cython
import numpy as np

cdef extern int __builtin_clzl(unsigned long x)


def leading_zeros_cython(unsigned long x):
    return __builtin_clzl(x)


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


cdef inline int findSplit(unsigned long[:] sortedMortonCodes, int first, int last) noexcept:
    # Identical Morton codes => split the range in the middle.

    cdef int commonPrefix, split, step, newSplit, splitPrefix
    cdef unsigned long firstCode, lastCode, splitCode

    firstCode = sortedMortonCodes[first]
    lastCode = sortedMortonCodes[last]

    if (firstCode == lastCode):
        return (first + last) >> 1

    # Calculate the number of highest bits that are the same
    # for all objects, using the count-leading-zeros intrinsic.

    commonPrefix = __builtin_clzl(firstCode ^ lastCode)

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
            splitPrefix = __builtin_clzl(firstCode ^ splitCode)
            if (splitPrefix > commonPrefix):
                split = newSplit  # accept proposal

    return split


cdef inline void determineRange(unsigned long[:] sortedMortonCodes, int n_codes, int idx, int[:] firstlast) noexcept:
    """
    The determine range function needed by the generateHierarchy function.
    This code has has been adapted from the CornerStone CUDA/C++ code,
    see constructInternalNode in btree.hpp.
    """

    cdef int d, minPrefixLength, firstIndex, max_last, lz_p1, lz_m1
    cdef int searchRange, secondIndex, lz_first_second, newJdx, lz_f_Jdx
    cdef int first, last
    cdef unsigned long p1, m1

    d = 1
    minPrefixLength = -1
    firstIndex = idx
    max_last = n_codes

    if (firstIndex > 0):
        p1 = sortedMortonCodes[firstIndex] ^ sortedMortonCodes[firstIndex + 1]
        m1 = sortedMortonCodes[firstIndex] ^ sortedMortonCodes[firstIndex - 1]

        # Count leading zeros
        lz_p1 = __builtin_clzl(p1)
        lz_m1 = __builtin_clzl(m1)

        if lz_p1 > lz_m1:
            d = 1
            minPrefixLength = lz_m1
        else:
            d = -1
            minPrefixLength = lz_p1

    searchRange = 2
    secondIndex = firstIndex + searchRange * d

    while (0 <= secondIndex and secondIndex < max_last):
        lz_first_second = __builtin_clzl(
            sortedMortonCodes[firstIndex] ^ sortedMortonCodes[secondIndex])
        if lz_first_second > minPrefixLength:
            searchRange *= 2
            secondIndex = firstIndex + searchRange * d
            if secondIndex < max_last:
                lz_first_second = __builtin_clzl(
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
            lz_f_Jdx = __builtin_clzl(
                sortedMortonCodes[firstIndex] ^ sortedMortonCodes[newJdx])
            if lz_f_Jdx > minPrefixLength:
                secondIndex = newJdx

    first = min(secondIndex, firstIndex)
    last = max(secondIndex, firstIndex)

    firstlast[0] = first
    firstlast[1] = last
    # return first, last


def generateHierarchy(unsigned long[:] sortedMortonCodes, long[:, :] tree_children, long[:] tree_parents):
    """
    Generate tree using the sorted Morton code.
    This is done in parallel, see this blogpost
    https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    and the corresponding paper:

    """
    cdef int n_codes, num_internal_nodes, idx, first, last, split
    cdef long childA, childB
    cdef int[:] firstlast = np.zeros(2, dtype=np.int32)

    n_codes = sortedMortonCodes.shape[0]
    num_internal_nodes = n_codes - 1

    # Launch thread for each internal node
    for idx in range(num_internal_nodes):

        # Find out which range of objects the node corresponds to.
        # (This is where the magic happens!)

        # Determine range
        determineRange(sortedMortonCodes, n_codes, idx, firstlast)
        first = firstlast[0]
        last = firstlast[1]

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


def propagate_bounds_upwards(unsigned long[:, :, :] tree_bounds, long[:] tree_parents,
                             int num_leafs, int num_internal_nodes):

    cdef int ip, child, next_parent
    cdef unsigned long x_min, x_max, y_min, y_max, z_min, z_max


    # For turning this loop openmp parallel, see here:
    # https://github.com/cython/cython/issues/3585

    # Loop over all leafs
    for ip in range(num_leafs):
        child = ip + num_internal_nodes
        next_parent = tree_parents[child]

        while next_parent != -1:  # indicates hitting root
            x_min = tree_bounds[child, 0, 0]
            y_min = tree_bounds[child, 1, 0]
            z_min = tree_bounds[child, 2, 0]
            x_max = tree_bounds[child, 0, 1]
            y_max = tree_bounds[child, 1, 1]
            z_max = tree_bounds[child, 2, 1]
            tree_bounds[next_parent, 0, 0] = min(tree_bounds[next_parent, 0, 0], x_min)
            tree_bounds[next_parent, 1, 0] = min(tree_bounds[next_parent, 1, 0], y_min)
            tree_bounds[next_parent, 2, 0] = min(tree_bounds[next_parent, 2, 0], z_min)
            tree_bounds[next_parent, 0, 1] = max(tree_bounds[next_parent, 0, 1], x_max)
            tree_bounds[next_parent, 1, 1] = max(tree_bounds[next_parent, 1, 1], y_max)
            tree_bounds[next_parent, 2, 1] = max(tree_bounds[next_parent, 2, 1], z_max)


            # Prepare next iteration of while loop
            child = next_parent
            next_parent = tree_parents[next_parent]
