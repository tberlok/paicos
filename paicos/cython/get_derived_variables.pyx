from cython.parallel import prange
import numpy as np
cimport numpy as np
cimport libc.math as math

ctypedef fused real_t:
    float
    double
    np.float32_t
    np.float64_t


def get_curvature(real_t[:, :] Bvec, real_t[:, :] Bgradient):

    """
    This function computes the magnetic field line curvature.

    Parameters:
        Bvec (array N, 3): Magnetic field vector
        Bgradient (array N, 3, 3): Magnetic field gradient tensor

    Returns:
        array N: The magnetic field line curvature.
    """
    cdef int ip, ii, jj, kk
    cdef int Np = Bvec.shape[0]
    cdef real_t B2, gradB_ij, gradB_kj, term1, term2

    cdef real_t[:] K2 = np.zeros(Np, dtype=np.float64)

    """
    The code is equivalent to
    Kvec  = np.dot(Bgrad_mat[i,],Bvec[i,]) / B[i]**2
    Kvec -= np.dot(np.dot(Bgrad_mat[i,],Bvec[i,]),Bvec[i,]) * Bvec[i,] / B[i]**4
    K2 = (Kvec**2).sum())
    K = np.sqrt(K2)
    """

    # Bgradient has not been reshaped, \nabla B_ij
    for ip in prange(Np, nogil=True, schedule='static'): #, num_threads=32):
        B2 = 0.0
        for ii in range(3):
            B2 = B2 + Bvec[ip, ii]**2
        for kk in range(3):
            term1 = 0.0
            term2 = 0.0
            for jj in range(3):
                gradB_kj = Bgradient[ip, kk*3+jj]
                term1 = term1 + gradB_kj*Bvec[ip, jj]
                for ii in range(3):
                    gradB_ij = Bgradient[ip, ii*3+jj]
                    term2 = term2 + gradB_ij*Bvec[ip, jj]*Bvec[ip, ii]*Bvec[ip, kk]

            K2[ip] = K2[ip] + (term1/B2 - term2/B2**2.0)**2.0

    # Return a numpy array instead of a view
    tmp = np.zeros(Np, dtype=np.float64)
    tmp[:] = np.sqrt(K2[:])
    return tmp


def get_magnitude_of_vector(real_t[:, :] Bvec):

    """
    Computes the magnitude of a vector, e.g., magnetic field strength.
    """
    cdef int ip, ii
    cdef int Np = Bvec.shape[0]
    cdef real_t B2

    cdef real_t[:] B = np.zeros(Np, dtype=np.float64)

    # Bgradient has not been reshaped, \nabla B_ij
    for ip in prange(Np, nogil=True, schedule='static'):
        B2 = 0.0
        for ii in range(3):
            B2 = B2 + Bvec[ip, ii]**2
        B[ip] = math.sqrt(B2)

    # Return a numpy array instead of a view
    tmp = np.zeros(Np, dtype=np.float64)
    tmp[:] = B[:]
    return tmp

def sum_1d_array_omp(real_t[:] arr, int num_threads):

    """
    Computes the sum of a 1D array using openmp.
    Equivalent to np.sum(arr)
    """
    cdef int ip
    cdef int Np = arr.shape[0]
    cdef real_t the_sum = 0.0

    for ip in prange(Np, nogil=True, schedule='static', num_threads=num_threads):
        the_sum += arr[ip]

    return the_sum

def sum_2d_array_omp(real_t[:, :] arr, int num_threads):

    """
    Computes the sum of an array along the first index using openmp.
    Equivalent to np.sum(arr, axis=0)
    """
    cdef int ip
    cdef int Np = arr.shape[0]
    cdef real_t jj_sum = 0.0

    the_sum = np.zeros(arr.shape[1])
    for jj in range(arr.shape[1]):
        jj_sum = 0.0
        for ip in prange(Np, nogil=True, schedule='static', num_threads=num_threads):
            jj_sum += arr[ip, jj]

        the_sum[jj] = jj_sum

    return the_sum

def sum_arr_times_vector_omp(real_t[:] arr, real_t[:, :] vector, int num_threads):

    """
    Computes the sum of an array times a vector.
    e.g. sum_i M_i vec{v}_i for calculating the center of mass.

    Equivalent to np.sum(arr[:, None] * vector, axis=0)
    """
    cdef int ip
    cdef int Np = arr.shape[0]
    cdef real_t the_sum_x = 0.0
    cdef real_t the_sum_y = 0.0
    cdef real_t the_sum_z = 0.0

    for ip in prange(Np, nogil=True, schedule='static', num_threads=num_threads):
        the_sum_x += arr[ip] * vector[ip, 0]
        the_sum_y += arr[ip] * vector[ip, 1]
        the_sum_z += arr[ip] * vector[ip, 2]

    return np.array([the_sum_x, the_sum_y, the_sum_z])

def sum_arr_times_vector_cross_product(real_t[:] mass, real_t[:, :] coord, real_t[:, :] velocity,
                                       real_t[:] center, int num_threads):

    """
    This code calculates sum_i (mass_i (coord_ij - center) x velocity_ij).

    That is, it returns the total angular momentum vector.
    """
    cdef int ip
    cdef int Np = mass.shape[0]
    cdef real_t the_sum_x = 0.0
    cdef real_t the_sum_y = 0.0
    cdef real_t the_sum_z = 0.0
    cdef real_t vx, vy, vz, rx, ry, rz

    for ip in prange(Np, nogil=True, schedule='static', num_threads=num_threads):
        vx = velocity[ip, 0]
        vy = velocity[ip, 1]
        vz = velocity[ip, 2]
        rx = coord[ip, 0] - center[0]
        ry = coord[ip, 1] - center[1]
        rz = coord[ip, 2] - center[2]
        the_sum_x += mass[ip] * (ry*vz - rz*vy)
        the_sum_y += mass[ip] * (rz*vx - rx*vz)
        the_sum_z += mass[ip] * (rx*vy - ry*vx)

    return np.array([the_sum_x, the_sum_y, the_sum_z])
