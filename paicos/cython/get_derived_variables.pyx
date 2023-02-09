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
    This function returns the curvature, equivalent
    to code:

    Kvec  = np.dot(Bgrad_mat[i,],Bvec[i,]) / B[i]**2
    Kvec -= np.dot(np.dot(Bgrad_mat[i,],Bvec[i,]),Bvec[i,]) * Bvec[i,] / B[i]**4
    K2 = (Kvec**2).sum())
    K = np.sqrt(K2)
    """
    cdef int ip, ii, jj, kk
    cdef int Np = Bvec.shape[0]
    cdef real_t B2, gradB_ij, gradB_kj, term1, term2

    cdef real_t[:] K2 = np.zeros(Np, dtype=np.float64)

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
    Compute the magnitude of a vector, e.g., magnetic field strength
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
