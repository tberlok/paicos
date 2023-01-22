from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free
cimport openmp
import cython
import numpy as np
cimport numpy as np

# ctypedef double real_t

ctypedef fused real_t:
    float
    double
    np.float32_t
    np.float64_t

STUFF = "Hi" # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu

# https://stackoverflow.com/questions/54776301/cython-prange-is-repeating-not-parallelizing

def get_openmp_settings(mpirank, verbose=True):
    """
    """

    cdef int maxthreads = openmp.omp_get_max_threads()
    if verbose:
        print('mpirank {} has maximum of {} threads'.format(mpirank, maxthreads))

    return maxthreads


cpdef double simple_reduction(int n, int num_threads):
    cdef int i
    cdef int sum = 0


    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        sum += 1
    return sum
