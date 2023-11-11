from cython.parallel import prange
cimport openmp
import numpy as np
cimport numpy as np

ctypedef fused real_t:
    float
    double
    np.float32_t
    np.float64_t


def get_cube(real_t [:, :] pos, real_t xc, real_t yc, real_t zc,
             real_t sidelength_x, real_t sidelength_y,
             real_t thickness, real_t boxsize, int numthreads):

    cdef int Np = pos.shape[0]
    cdef int ip
    cdef real_t x, y, z

    cdef int[:] index = np.zeros(Np, dtype=np.intc)

    openmp.omp_set_num_threads(numthreads)

    for ip in prange(Np, nogil=True, schedule='static'):
        x = pos[ip, 0] - xc
        y = pos[ip, 1] - yc
        z = pos[ip, 2] - zc

        # box_wrap_diff function
        if x < -0.5*boxsize:
            x = x + boxsize
        elif x > 0.5*boxsize:
            x = x - boxsize

        if y < -0.5*boxsize:
            y = y + boxsize
        elif y > 0.5*boxsize:
            y = y - boxsize

        if z < -0.5*boxsize:
            z = z + boxsize
        elif z > 0.5*boxsize:
            z = z - boxsize

        # Index calculation
        index[ip] = 0
        if (x < sidelength_x/2.0) and (x > -sidelength_x/2):
            if (y < sidelength_y/2.0) and (y > -sidelength_y/2):
                if (z > -0.5*thickness):
                    if (z < 0.5*thickness):
                        index[ip] = 1

    # Return a numpy boolean array
    tmp = np.zeros(Np, dtype=np.bool_)
    tmp[:] = index[:]
    return tmp


def get_cube_plus_thin_layer(real_t [:, :] pos, real_t xc, real_t yc, real_t zc,
                             real_t sidelength_x, real_t sidelength_y,
                             real_t sidelength_z, real_t [:] thickness,
                             real_t boxsize, int numthreads):

    cdef int Np = pos.shape[0]
    cdef int ip
    cdef real_t x, y, z

    cdef int[:] index = np.zeros(Np, dtype=np.intc)

    openmp.omp_set_num_threads(numthreads)

    for ip in prange(Np, nogil=True, schedule='static'):
        x = pos[ip, 0] - xc
        y = pos[ip, 1] - yc
        z = pos[ip, 2] - zc

        # box_wrap_diff function
        if x < -0.5*boxsize:
            x = x + boxsize
        elif x > 0.5*boxsize:
            x = x - boxsize

        if y < -0.5*boxsize:
            y = y + boxsize
        elif y > 0.5*boxsize:
            y = y - boxsize

        if z < -0.5*boxsize:
            z = z + boxsize
        elif z > 0.5*boxsize:
            z = z - boxsize

        # Index calculation
        index[ip] = 0
        if (x < (sidelength_x + thickness[ip])/2.0) and (x > -(sidelength_x + thickness[ip])/2.0):
            if (y < (sidelength_y + thickness[ip])/2.0) and (y > -(sidelength_y + thickness[ip])/2.0):
                if (z < (sidelength_z + thickness[ip])/2.0):
                    if (z > -(sidelength_z + thickness[ip])/2.0):
                        index[ip] = 1

    # Return a numpy boolean array
    tmp = np.zeros(Np, dtype=np.bool_)
    tmp[:] = index[:]
    return tmp

def get_rotated_cube(real_t [:, :] pos, real_t xc, real_t yc, real_t zc,
                             real_t sidelength_x, real_t sidelength_y,
                             real_t sidelength_z,
                             real_t boxsize,
                             real_t[:] unit_vector_x,
                             real_t[:] unit_vector_y,
                             real_t[:] unit_vector_z,
                             int numthreads):

    cdef int Np = pos.shape[0]
    cdef int ip
    cdef real_t x, y, z
    cdef real_t x_dot_ex, x_dot_ey, x_dot_ez

    cdef int[:] index = np.zeros(Np, dtype=np.intc)

    openmp.omp_set_num_threads(numthreads)

    for ip in prange(Np, nogil=True, schedule='static'):
        x = pos[ip, 0] - xc
        y = pos[ip, 1] - yc
        z = pos[ip, 2] - zc

        # box_wrap_diff function
        if x < -0.5*boxsize:
            x = x + boxsize
        elif x > 0.5*boxsize:
            x = x - boxsize

        if y < -0.5*boxsize:
            y = y + boxsize
        elif y > 0.5*boxsize:
            y = y - boxsize

        if z < -0.5*boxsize:
            z = z + boxsize
        elif z > 0.5*boxsize:
            z = z - boxsize

        x_dot_ex = x * unit_vector_x[0] + y * unit_vector_x[1] + z * unit_vector_x[2]
        x_dot_ey = x * unit_vector_y[0] + y * unit_vector_y[1] + z * unit_vector_y[2]
        x_dot_ez = x * unit_vector_z[0] + y * unit_vector_z[1] + z * unit_vector_z[2]

        # Index calculation
        index[ip] = 0
        if (x_dot_ex < sidelength_x/2.0) and (x_dot_ex > -sidelength_x/2.0):
            if (x_dot_ey < sidelength_y/2.0) and (x_dot_ey > -sidelength_y/2.0):
                if (x_dot_ez < sidelength_z/2.0):
                    if (x_dot_ez > -sidelength_z/2.0):
                        index[ip] = 1

    # Return a numpy boolean array
    tmp = np.zeros(Np, dtype=np.bool_)
    tmp[:] = index[:]
    return tmp

def get_rotated_cube_plus_thin_layer(real_t [:, :] pos, real_t xc, real_t yc, real_t zc,
                             real_t sidelength_x, real_t sidelength_y,
                             real_t sidelength_z, real_t [:] thickness,
                             real_t boxsize,
                             real_t[:] unit_vector_x,
                             real_t[:] unit_vector_y,
                             real_t[:] unit_vector_z,
                             int numthreads):

    cdef int Np = pos.shape[0]
    cdef int ip
    cdef real_t x, y, z
    cdef real_t x_dot_ex, x_dot_ey, x_dot_ez

    cdef int[:] index = np.zeros(Np, dtype=np.intc)

    openmp.omp_set_num_threads(numthreads)

    for ip in prange(Np, nogil=True, schedule='static'):
        x = pos[ip, 0] - xc
        y = pos[ip, 1] - yc
        z = pos[ip, 2] - zc

        # box_wrap_diff function
        if x < -0.5*boxsize:
            x = x + boxsize
        elif x > 0.5*boxsize:
            x = x - boxsize

        if y < -0.5*boxsize:
            y = y + boxsize
        elif y > 0.5*boxsize:
            y = y - boxsize

        if z < -0.5*boxsize:
            z = z + boxsize
        elif z > 0.5*boxsize:
            z = z - boxsize

        x_dot_ex = x * unit_vector_x[0] + y * unit_vector_x[1] + z * unit_vector_x[2]
        x_dot_ey = x * unit_vector_y[0] + y * unit_vector_y[1] + z * unit_vector_y[2]
        x_dot_ez = x * unit_vector_z[0] + y * unit_vector_z[1] + z * unit_vector_z[2]

        # Index calculation
        index[ip] = 0
        if (x_dot_ex < (sidelength_x + thickness[ip])/2.0) and (x_dot_ex > -(sidelength_x + thickness[ip])/2.0):
            if (x_dot_ey < (sidelength_y + thickness[ip])/2.0) and (x_dot_ey > -(sidelength_y + thickness[ip])/2.0):
                if (x_dot_ez < (sidelength_z + thickness[ip])/2.0):
                    if (x_dot_ez > -(sidelength_z + thickness[ip])/2.0):
                        index[ip] = 1

    # Return a numpy boolean array
    tmp = np.zeros(Np, dtype=np.bool_)
    tmp[:] = index[:]
    return tmp


def get_radial_range(real_t [:, :] pos, real_t xc, real_t yc,
                    real_t zc, real_t r_min, real_t r_max,
                    int numthreads):

    cdef int Np = pos.shape[0]
    cdef int ip
    cdef real_t x, y, z, r2

    cdef real_t r2_min = r_min*r_min
    cdef real_t r2_max = r_max*r_max

    cdef int[:] index = np.zeros(Np, dtype=np.intc)

    openmp.omp_set_num_threads(numthreads)

    for ip in prange(Np, nogil=True, schedule='static'):
        x = pos[ip, 0] - xc
        y = pos[ip, 1] - yc
        z = pos[ip, 2] - zc

        r2 = x*x + y*y + z*z

        # Index calculation
        index[ip] = 0
        if (r2 < r2_max) and (r2 > r2_min):
            index[ip] = 1

    # Return a numpy boolean array
    tmp = np.zeros(Np, dtype=np.bool_)
    tmp[:] = index[:]
    return tmp
