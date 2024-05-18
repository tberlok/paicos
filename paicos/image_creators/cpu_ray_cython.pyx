from cython.parallel import prange, parallel
cimport openmp
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

cdef inline bint is_point_in_box(double[:, :] point, unsigned long[:, :] box, int threadnum) noexcept nogil:
    cdef int ii
    for ii in range(3):
        if point[threadnum, ii] < box[ii, 0] or point[threadnum, ii] > box[ii, 1]:
            return False
    return True


cdef inline double distance(double[:] point, double[:, :] query_point, int threadnum) noexcept nogil:
    dist = sqrt((point[0] - query_point[threadnum, 0])**2
             + (point[1] - query_point[threadnum, 1])**2
             + (point[2] - query_point[threadnum, 2])**2)
    return dist

cdef inline int nearest_neighbor_device(double[:, :] points, long[: ]tree_parents,
            long[:,:] tree_children, unsigned long[:,:,:] tree_bounds,
                            double[:, :] query_point, int num_internal_nodes,
                            long[:, :] queue, int threadnum) noexcept nogil:

    cdef int L = 21
    cdef int queue_index, min_index, node_id
    cdef unsigned long childA, childB
    cdef bint is_leafA, is_leafB, point_in_A, point_in_B, traverseA, traverseB
    cdef double min_dist

    # We traverse the nodes and leafs using a while loop and a queue.
    # Local memory on each tread (32 should be fine?)
    # queue = cuda.local.array(128, numba.int64)
    # cdef long[:] queue = np.zeros(128, dtype=np.int64)
    # Initialize queue_index an start at node 0
    queue_index = 0
    queue[threadnum, queue_index] = 0

    # Initialize min_dist, and min_index
    min_dist = (2**L - 1.0)
    min_index = -1

    while queue_index >= 0:

        node_id = queue[threadnum, queue_index]
        # print(queue_index, node_id)#traverseA, traverseB, is_leafA, is_leafB)

        childA = tree_children[node_id, 0]
        childB = tree_children[node_id, 1]

        is_leafA = childA >= num_internal_nodes
        is_leafB = childB >= num_internal_nodes

        point_in_A = is_point_in_box(query_point, tree_bounds[childA], threadnum)
        point_in_B = is_point_in_box(query_point, tree_bounds[childB], threadnum)

        # Do explicit check if in a leaf
        if point_in_A and is_leafA:
            data_id = childA - num_internal_nodes
            dist = distance(points[data_id], query_point, threadnum)

            if dist < min_dist:
                min_dist = dist
                min_index = data_id

        if point_in_B and is_leafB:
            data_id = childB - num_internal_nodes
            dist = distance(points[data_id], query_point, threadnum)

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
                queue[threadnum, queue_index] = childA
            else:
                queue[threadnum, queue_index] = childB
            if traverseA and traverseB:
                queue_index += 1
                queue[threadnum, queue_index] = childB

    return min_index

cdef inline void rotate_point_around_center(double[:, :] point, double[:, :] tmp_point,
                            double[:] center, double[:, :] rotation_matrix, int threadnum) noexcept nogil:
    """
    Rotate around center. Note that we overwrite point.
    """
    cdef int ii, jj

    # Subtract center
    for ii in range(3):
        tmp_point[threadnum, ii] = point[threadnum, ii] - center[ii]
        point[threadnum, ii] = 0.0

    # Rotate around center (matrix multiplication).
    for ii in range(3):
        for jj in range(3):
            point[threadnum, ii] += rotation_matrix[ii, jj] * tmp_point[threadnum, jj]

    # Add center back
    for ii in range(3):
        point[threadnum, ii] = point[threadnum, ii] + center[ii]


def trace_rays(double[:, :] points, long[:] tree_parents,
               long[:, :] tree_children, unsigned long[:, :, :] tree_bounds,
               double[:] variable, double[:] hsml,
               double[:] widths, double[:] center,
               double tree_scale_factor, double[:] tree_offsets,
               double[:, :] image, double[:, :] rotation_matrix,
               double tol, int numthreads=1):

    cdef int num_internal_nodes, min_index

    cdef int ip, ix, iy, nx, ny, threadnum

    cdef double dx, dy, dz, z, result
    nx = image.shape[0]
    ny = image.shape[1]

    cdef int Np = nx * ny

    # Making this openmp parallel requires extending the
    # three arrays below to have an additional dimension with the number of threads
    cdef double[:, :] query_point = np.zeros((numthreads, 3), dtype=np.float64)
    cdef double[:, :] tmp_point = np.zeros((numthreads, 3), dtype=np.float64)
    cdef long[:, :] queue = np.zeros((numthreads, 128), dtype=np.int64)

    # Could do better memory-wise by letting each thread do a smaller array...
    # cdef double[:] tmp_var = np.zeros(Np, dtype=np.float64)

    dx = widths[0] / nx
    dy = widths[1] / ny

    num_internal_nodes = tree_children.shape[0]

    # with nogil, parallel(num_threads=numthreads):
    for ip in prange(Np, nogil=True, schedule='static', num_threads=numthreads):
        threadnum = openmp.omp_get_thread_num()
        result = 0.0

        ix = ip // ny
        iy = ip - ix * ny

        # Initialize z
        z = 0.0

        while z < widths[2]:

            # Query points in aligned coords
            query_point[threadnum, 0] = (center[0] - widths[0] / 2.0) + (ix + 0.5) * dx
            query_point[threadnum, 1] = (center[1] - widths[1] / 2.0) + (iy + 0.5) * dy
            query_point[threadnum, 2] = (center[2] - widths[2] / 2.0) + z

            # Rotate to simulation coords
            rotate_point_around_center(
                query_point, tmp_point, center, rotation_matrix, threadnum)

            # Convert to the tree coordinates
            query_point[threadnum, 2] = (query_point[threadnum, 2] - tree_offsets[2]
                              ) * tree_scale_factor
            query_point[threadnum, 0] = (query_point[threadnum, 0] - tree_offsets[0]
                              ) * tree_scale_factor
            query_point[threadnum, 1] = (query_point[threadnum, 1] - tree_offsets[1]
                              ) * tree_scale_factor

            min_index = nearest_neighbor_device(points, tree_parents, tree_children,
                                                tree_bounds, query_point,
                                                num_internal_nodes, queue, threadnum)

            # Calculate dz
            dz = tol * hsml[min_index]

            # Update integral
            result = result + dz * variable[min_index]

            # Update position
            z = z + dz

        # Subtract the 'extra' stuff added in last iteration
        result = result - (z - widths[2]) * variable[min_index]
        # Set result in image array
        image[ix, iy] = result
