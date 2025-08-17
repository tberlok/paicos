from cython.parallel import prange, parallel
from libc.math cimport fmin, fmax, sqrt
from libc.stdlib cimport malloc, free
cimport openmp
import numpy as np
cimport numpy as np

ctypedef double real_t

cdef inline int rotate_point_around_center(real_t[:] point,
                                            real_t[:] tmp_point,
                                            real_t[:] center,
                                            real_t[:, :] rotation_matrix) noexcept nogil:
    cdef int ii, jj

    # Subtract center and zero out point
    for ii in range(3):
        tmp_point[ii] = point[ii] - center[ii]
        point[ii] = 0.0

    # Matrix-vector multiplication
    for ii in range(3):
        for jj in range(3):
            point[ii] += rotation_matrix[ii, jj] * tmp_point[jj]

    # Add center back
    for ii in range(3):
        point[ii] += center[ii]

    return 0


cdef inline real_t distance(real_t[:] p1, real_t[:] p2) noexcept nogil:
    cdef real_t dx = p1[0] - p2[0]
    cdef real_t dy = p1[1] - p2[1]
    cdef real_t dz = p1[2] - p2[2]
    return sqrt(dx*dx + dy*dy + dz*dz)

cdef inline real_t distance_to_box(real_t[:] point, unsigned long[:, :] box) noexcept nogil:
    cdef real_t d
    cdef real_t sq_dist = 0.0
    cdef int i
    for i in range(3):
        if point[i] < box[i, 0]:
            d = box[i, 0] - point[i]
            sq_dist += d * d
        elif point[i] > box[i, 1]:
            d = point[i] - box[i, 1]
            sq_dist += d * d
    return sqrt(sq_dist)

cdef inline int nearest_neighbor(real_t[:, :] points, long[:] tree_parents,
                                   long[:, :] tree_children, unsigned long[:, :, :] tree_bounds,
                                   real_t[:] query_point, int num_internal_nodes, real_t[:] min_dist_final) noexcept nogil:
    cdef int queue[128]
    cdef int queue_index = 0
    queue[queue_index] = 0
    cdef int L = 21
    cdef real_t min_dist = (2 ** L - 1.0)
    cdef int min_index = -1
    cdef int node_id, childA, childB, data_id
    cdef bint is_leafA, is_leafB
    cdef real_t dist, distA, distB
    cdef bint traverseA, traverseB

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
    min_dist_final[0] = min_dist
    return min_index

cdef inline bint is_point_in_box(real_t[:] point, unsigned long[:, :] box) noexcept nogil:
    cdef int ii
    for ii in range(3):
        if point[ii] < box[ii, 0] or point[ii] > box[ii, 1]:
            return False
    return True

cdef inline int nearest_neighbor_voronoi(real_t[:, :] points, long[:] tree_parents,
                                   long[:, :] tree_children, unsigned long[:, :, :] tree_bounds,
                                   real_t[:] query_point, int num_internal_nodes, real_t[:] min_dist_final) noexcept nogil:
    cdef int queue[128]
    cdef int queue_index = 0
    queue[queue_index] = 0
    cdef int L = 21
    cdef real_t min_dist = (2 ** L - 1.0)
    cdef int min_index = -1
    cdef int node_id, childA, childB, data_id
    cdef bint is_leafA, is_leafB, point_in_A, point_in_B
    cdef real_t dist, distA, distB
    cdef bint traverseA, traverseB

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
    min_dist_final[0] = min_dist
    return min_index

# from cython.parallel cimport prange, parallel
# from libc.math cimport fmin, fmax

# # Assume real_t is already defined, e.g.
# # ctypedef double real_t

# from cython.parallel cimport prange, parallel
# from libc.math cimport sqrt
# cimport cython

# # Assume: real_t is already defined (e.g., ctypedef double real_t)
# # Also assume: nearest_neighbor() and rotate_point_around_center() are declared elsewhere and cimported

def trace_rays_cpu(real_t[:, :] points,
                   long[:] tree_parents,
                   long[:, :] tree_children,
                   unsigned long[:, :, :] tree_bounds,
                   real_t[:] variable,
                   real_t[:] hsml,
                   real_t[:] widths,
                   real_t[:] center,
                   real_t tree_scale_factor,
                   real_t[:] tree_offsets,
                   real_t[:, :] image,
                   real_t[:, :] rotation_matrix,
                   real_t tol,
                   int numthreads):

    cdef int nx = image.shape[0]
    cdef int ny = image.shape[1]
    cdef real_t dx = widths[0] / nx
    cdef real_t dy = widths[1] / ny

    cdef int ix, iy, i
    cdef real_t z, dz, result
    cdef real_t min_dist
    cdef int min_index

    cdef int num_internal_nodes = tree_children.shape[0]

    cdef real_t[:] query_point = np.zeros((3), dtype=np.float64)
    cdef real_t[:] tmp_point = np.zeros((3), dtype=np.float64)
    cdef real_t[:] min_dist_final = np.zeros((1), dtype=np.float64)

    # with nogil, parallel(num_threads=numthreads):
    # for ix in range(nx, schedule='static'):
    for ix in range(nx):
        for iy in range(ny):
            result = 0.0
            z = 0.0

            while z < widths[2]:
                # Compute query point in world coordinates
                query_point[0] = (center[0] - widths[0] / 2.0) + (ix + 0.5) * dx
                query_point[1] = (center[1] - widths[1] / 2.0) + (iy + 0.5) * dy
                query_point[2] = (center[2] - widths[2] / 2.0) + z

                # Rotate point around center (in-place)
                rotate_point_around_center(query_point, tmp_point, center, rotation_matrix)

                # Transform to tree coordinates
                for i in range(3):
                    query_point[i] = (query_point[i] - tree_offsets[i]) * tree_scale_factor

                # Nearest neighbor
                min_index = nearest_neighbor(points, tree_parents, tree_children,
                                            tree_bounds, query_point, num_internal_nodes, min_dist_final)

                # min_dist = distance(points[min_index], query_point)
                min_dist = min_dist_final[0]

                # Adaptive integration step (convert to Arepo units)
                dz = tol * min_dist / tree_scale_factor
                result = result + dz * variable[min_index]
                z = z + dz

            # Overshoot correction
            result -= (z - widths[2]) * variable[min_index]
            image[ix, iy] = result


def trace_rays_cpu_voronoi(real_t[:, :] points,
                   long[:] tree_parents,
                   long[:, :] tree_children,
                   unsigned long[:, :, :] tree_bounds,
                   real_t[:] variable,
                   real_t[:] hsml,
                   real_t[:] widths,
                   real_t[:] center,
                   real_t tree_scale_factor,
                   real_t[:] tree_offsets,
                   real_t[:, :] image,
                   real_t[:, :] rotation_matrix,
                   real_t tol,
                   int numthreads):

    cdef int nx = image.shape[0]
    cdef int ny = image.shape[1]
    cdef real_t dx = widths[0] / nx
    cdef real_t dy = widths[1] / ny

    cdef int ix, iy, i
    cdef real_t z, dz, result
    cdef real_t min_dist
    cdef int min_index

    cdef int num_internal_nodes = tree_children.shape[0]

    cdef real_t[:] query_point = np.zeros((3), dtype=np.float64)
    cdef real_t[:] tmp_point = np.zeros((3), dtype=np.float64)
    cdef real_t[:] min_dist_final = np.zeros((1), dtype=np.float64)

    # with nogil, parallel(num_threads=numthreads):
    # for ix in range(nx, schedule='static'):
    for ix in range(nx):
        for iy in range(ny):
            result = 0.0
            z = 0.0

            while z < widths[2]:
                # Compute query point in world coordinates
                query_point[0] = (center[0] - widths[0] / 2.0) + (ix + 0.5) * dx
                query_point[1] = (center[1] - widths[1] / 2.0) + (iy + 0.5) * dy
                query_point[2] = (center[2] - widths[2] / 2.0) + z

                # Rotate point around center (in-place)
                rotate_point_around_center(query_point, tmp_point, center, rotation_matrix)

                # Transform to tree coordinates
                for i in range(3):
                    query_point[i] = (query_point[i] - tree_offsets[i]) * tree_scale_factor

                # Nearest neighbor
                min_index = nearest_neighbor_voronoi(points, tree_parents, tree_children,
                                            tree_bounds, query_point, num_internal_nodes, min_dist_final)
                if min_index == -1:
                    min_index = nearest_neighbor(points, tree_parents, tree_children,
                                            tree_bounds, query_point, num_internal_nodes, min_dist_final)


                # min_dist = distance(points[min_index], query_point)
                min_dist = min_dist_final[0]

                # Adaptive integration step (convert to Arepo units)
                dz = tol * hsml[min_index]
                result = result + dz * variable[min_index]
                z = z + dz

            # Overshoot correction
            result -= (z - widths[2]) * variable[min_index]
            image[ix, iy] = result
