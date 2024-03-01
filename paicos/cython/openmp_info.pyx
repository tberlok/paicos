from cython.parallel import prange
cimport openmp
import numpy as np
cimport numpy as np

# ctypedef double real_t

ctypedef fused real_t:
    float
    double
    np.float32_t
    np.float64_t

STUFF = "Hi"  # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu

# https://stackoverflow.com/questions/54776301/cython-prange-is-repeating-not-parallelizing


def get_openmp_settings(mpirank, verbose=True):
    """
    This helper function returns the maximum number of
    openmp threads available on the system.
    """

    cdef int maxthreads = openmp.omp_get_max_threads()
    if verbose:
        print('mpirank {} has maximum of {} threads'.format(mpirank, maxthreads))

    return maxthreads


cpdef double simple_reduction(int n, int num_threads):
    """
    This program is a simple reduction using openmp parallel
    code. The return value should simply be identical to the input, n.
    Used to find a bug cython/openmp bug that at some point occured on MacOs.
    """
    cdef int i
    cdef int mysum = 0

    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        mysum += 1
    return mysum
