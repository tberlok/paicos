import cython
import numpy as np
cimport numpy as np

ctypedef fused real_t:
    float
    double
    np.float32_t
    np.float64_t

def get_hist_from_weights_and_idigit(int num_bins, real_t[:] weights,
                                     long[:] i_digit):

    cdef int Np = weights.shape[0]

    cdef real_t[:] hist = np.zeros(num_bins+1, dtype=np.float64)

    cdef int ip, ib
    for ip in range(Np):
        ib = i_digit[ip]
        hist[ib] = hist[ib] + weights[ip]

    # Return a numpy array instead of a view
    tmp = np.zeros(num_bins-1, dtype=np.float64)
    tmp[:] = hist[1:num_bins]
    return tmp
