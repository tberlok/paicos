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
