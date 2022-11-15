from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free
cimport openmp
import cython
import numpy as np
cimport numpy as np

cimport libc.math as math

# ctypedef double real_t

ctypedef fused real_t:
    float
    double
    np.float32_t
    np.float64_t

STUFF = "Hi" # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu

# https://stackoverflow.com/questions/54776301/cython-prange-is-repeating-not-parallelizing

def print_openmp_settings(mpirank):
    """
    """

    cdef int maxthreads = openmp.omp_get_max_threads()
    print('mpirank {} has maximum of {} threads'.format(mpirank, maxthreads))

def wrap_pos_and_get_index_of_region(real_t [:, :] pos, int[:] index,
                                     real_t xc, real_t yc, real_t zc,
                                     real_t sidelength, real_t thickness,
                                     real_t boxsize):

    cdef int Np = pos.shape[0]
    cdef int ip
    cdef real_t x, y, z

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
        if (z > -0.5*thickness):
            if (z < 0.5*thickness):
                index[ip] = 1

        pos[ip, 0] = x
        pos[ip, 1] = y
        pos[ip, 2] = z

def get_index_of_region(real_t [:, :] pos, real_t xc, real_t yc, real_t zc,
                        real_t sidelength_x, real_t sidelength_y,
                        real_t thickness, real_t boxsize):

    cdef int Np = pos.shape[0]
    cdef int ip
    cdef real_t x, y, z

    cdef int[:] index = np.zeros(Np, dtype=np.intc)

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

def get_index_of_x_slice_region(real_t [:, :] pos, real_t xc, real_t yc, real_t zc,
                        real_t sidelength_y, real_t sidelength_z,
                        real_t [:] thickness, real_t boxsize):

    cdef int Np = pos.shape[0]
    cdef int ip
    cdef real_t x, y, z

    cdef int[:] index = np.zeros(Np, dtype=np.intc)

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
        if (y < sidelength_y/2.0) and (y > -sidelength_y/2.0):
            if (z < sidelength_z/2.0) and (z > -sidelength_z/2.0):
                if (x > -0.5*thickness[ip]):
                    if (x < 0.5*thickness[ip]):
                        index[ip] = 1

    # Return a numpy boolean array
    tmp = np.zeros(Np, dtype=np.bool_)
    tmp[:] = index[:]
    return tmp

def get_index_of_y_slice_region(real_t [:, :] pos, real_t xc, real_t yc, real_t zc,
                        real_t sidelength_x, real_t sidelength_z,
                        real_t [:] thickness, real_t boxsize):

    cdef int Np = pos.shape[0]
    cdef int ip
    cdef real_t x, y, z

    cdef int[:] index = np.zeros(Np, dtype=np.intc)

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
        if (x < sidelength_x/2.0) and (x > -sidelength_x/2.0):
            if (z < sidelength_z/2.0) and (z > -sidelength_z/2.0):
                if (y > -0.5*thickness[ip]):
                    if (y < 0.5*thickness[ip]):
                        index[ip] = 1

    # Return a numpy boolean array
    tmp = np.zeros(Np, dtype=np.bool_)
    tmp[:] = index[:]
    return tmp

def get_index_of_z_slice_region(real_t [:, :] pos, real_t xc, real_t yc, real_t zc,
                        real_t sidelength_x, real_t sidelength_y,
                        real_t [:] thickness, real_t boxsize):

    cdef int Np = pos.shape[0]
    cdef int ip
    cdef real_t x, y, z

    cdef int[:] index = np.zeros(Np, dtype=np.intc)

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
        if (x < sidelength_x/2.0) and (x > -sidelength_x/2.0):
            if (y < sidelength_y/2.0) and (y > -sidelength_y/2.0):
                if (z > -0.5*thickness[ip]):
                    if (z < 0.5*thickness[ip]):
                        index[ip] = 1

    # Return a numpy boolean array
    tmp = np.zeros(Np, dtype=np.bool_)
    tmp[:] = index[:]
    return tmp

def box_wrap_x_and_y(real_t [:] x, real_t [:] y, real_t xc, real_t yc,
                     real_t boxsize):

    cdef int Np = x.shape[0]
    cdef int ip

    for ip in prange(Np, nogil=True, schedule='static'):
        x[ip] = x[ip] - xc
        y[ip] = y[ip] - yc

        # box_wrap_diff function
        if x[ip] < -0.5*boxsize:
            x[ip] = x[ip] + boxsize
        elif x[ip] > 0.5*boxsize:
            x[ip] = x[ip] - boxsize

        if y[ip] < -0.5*boxsize:
            y[ip] = y[ip] + boxsize
        elif y[ip] > 0.5*boxsize:
            y[ip] = y[ip] - boxsize

def project_image(real_t[:] xvec, real_t[:] yvec, real_t[:] variable,
                  real_t[:] hvec, int nx, real_t xc, real_t yc,
                  real_t sidelength_x, real_t sidelength_y,
                  real_t boxsize, int numthreads=1):

    """

    xc, yc are the coordinates at the center of a square image
    with sidelength=sidelength.
    This function assumes that only particles with coordinates

    x0 < x < x0 + sidelength_x
    y0 < y < y0 + sidelength_y

    are passed to it. For speed, this is not checked!

    Here x0, y0 are the coordinates at the lower, left corner.
    """

    assert numthreads == 1, 'use project_image_omp for more than one thread'

    # Number of particles
    cdef int Np = xvec.shape[0]

    # Shape of projection array
    cdef int ny = <int> (sidelength_y/sidelength_x * nx)
    assert (sidelength_y/sidelength_x * nx) == <float> ny, '(sidelength_y/sidelength_x * nx) needs to be an integer'

    # Create projection array
    cdef real_t[:,:] projection = np.zeros((nx, ny), dtype=np.float64)

    # Loop integers and other variables
    cdef int ip, ix, iy, ih
    cdef int ipx, ipy
    cdef int ix_min, ix_max
    cdef int iy_min, iy_max
    cdef real_t tx, ty, dx, dy, r2, h, h2, weight
    cdef real_t x, y, norm
    cdef real_t x0, y0

    assert sidelength_x/nx == sidelength_y/ny

    # Lower left corner of image in Arepo coordinates
    x0 = xc - sidelength_x/2.0
    y0 = yc - sidelength_y/2.0

    for ip in range(Np):
        # Center coordinate system at x0, y0
        x = xvec[ip] - x0
        y = yvec[ip] - y0

        # Apply periodic boundary condition
        if x < 0.0:
            x = x + boxsize
        elif x > boxsize:
            x = x - boxsize

        if y < 0.0:
            y = y + boxsize
        elif y > boxsize:
            y = y - boxsize

        # Position of particle in units of sidelength (0, sidelength)
        x = x*nx/sidelength_x
        y = y*ny/sidelength_y
        h = hvec[ip]*nx/sidelength_x

        if h < 1.:
            h = 1.

        # Index of closest grid point
        ipx = <int> x
        ipy = <int> y

        # Smoothing length as integer
        ih = <int> h + 1

        # Square of smoothing length
        h2 = h*h

        # Find minimum and maximum integers
        ix_min = ipx - ih
        iy_min = ipy - ih
        ix_max = ipx + ih
        iy_max = ipy + ih

        norm = 0.0
        for ix in range(ix_min, ix_max):
            for iy in range(iy_min, iy_max):
                dx = x - 0.5 - <real_t> ix
                dy = y - 0.5 - <real_t> iy
                r2 = dx*dx + dy*dy

                weight = 1.0 - r2/h2
                if weight > 0.0:
                    norm += weight

        # Find minimum and maximum integers
        ix_min = max(0, ipx - ih)
        iy_min = max(0, ipy - ih)
        ix_max = min(projection.shape[0], ipx + ih)
        iy_max = min(projection.shape[1], ipy + ih)

        for ix in range(ix_min, ix_max):
            for iy in range(iy_min, iy_max):
                dx = x - 0.5 - <real_t> ix
                dy = y - 0.5 - <real_t> iy
                r2 = dx*dx + dy*dy

                weight = 1.0 - r2/h2
                if weight > 0.0:
                    projection[ix, iy] += weight*variable[ip]/norm

    # Fix to avoid returning a memory-view
    tmp = np.zeros((nx, ny), dtype=np.float64)
    tmp[:, :] = projection[:, :]
    return tmp

def project_image_omp(real_t[:] xvec, real_t[:] yvec, real_t[:] variable,
                  real_t[:] hvec, int nx, real_t xc, real_t yc,
                  real_t sidelength_x, real_t sidelength_y,
                  real_t boxsize, int numthreads):

    # Number of particles
    cdef int Np = xvec.shape[0]

    # Shape of projection array
    cdef int ny = <int> (sidelength_y/sidelength_x * nx)

    assert (sidelength_y/sidelength_x * nx) == <float> ny, '(sidelength_y/sidelength_x * nx) needs to be an integer'

    # Loop integers and other variables
    cdef int ip, ix, iy, ih
    cdef int ipx, ipy
    cdef int ix_min, ix_max
    cdef int iy_min, iy_max
    cdef real_t tx, ty, dx, dy, r2, h, h2, weight
    cdef real_t x, y, norm
    cdef real_t x0, y0

    assert sidelength_x/nx == sidelength_y/ny

    # Lower left corner of image in Arepo coordinates
    x0 = xc - sidelength_x/2.0
    y0 = yc - sidelength_y/2.0

    cdef int threadnum, maxthreads

    maxthreads = openmp.omp_get_max_threads()

    cdef real_t[:,:, :] tmp_variable = np.zeros((nx, ny, numthreads), dtype=np.float64)
    cdef real_t[:,:] projection = np.zeros((nx, ny), dtype=np.float64)

    with nogil, parallel(num_threads=numthreads):
        for ip in prange(Np, schedule='static'):
            threadnum = openmp.omp_get_thread_num()
            # Center coordinate system at x0, y0
            x = xvec[ip] - x0
            y = yvec[ip] - y0

            # Apply periodic boundary condition
            if x < 0.0:
                x = x + boxsize
            elif x > boxsize:
                x = x - boxsize

            if y < 0.0:
                y = y + boxsize
            elif y > boxsize:
                y = y - boxsize

            # Position of particle in units of sidelength (0, sidelength)
            x = x*nx/sidelength_x
            y = y*ny/sidelength_y
            h = hvec[ip]*nx/sidelength_x

            if h < 1.:
                h = 1.

            # Index of closest grid point
            ipx = <int> x
            ipy = <int> y

            # Smoothing length as integer
            ih = <int> h + 1

            # Square of smoothing length
            h2 = h*h

            # Find minimum and maximum integers
            ix_min = ipx - ih
            iy_min = ipy - ih
            ix_max = ipx + ih
            iy_max = ipy + ih

            norm = 0.0
            for ix in range(ix_min, ix_max):
                for iy in range(iy_min, iy_max):
                    dx = x - 0.5 - <real_t> ix
                    dy = y - 0.5 - <real_t> iy
                    r2 = dx*dx + dy*dy

                    weight = 1.0 - r2/h2
                    if weight > 0.0:
                        norm = norm + weight

            # Find minimum and maximum integers
            ix_min = max(0, ipx - ih)
            iy_min = max(0, ipy - ih)
            ix_max = min(projection.shape[0], ipx + ih)
            iy_max = min(projection.shape[1], ipy + ih)

            for ix in range(ix_min, ix_max):
                for iy in range(iy_min, iy_max):
                    dx = x - 0.5 - <real_t> ix
                    dy = y - 0.5 - <real_t> iy
                    r2 = dx*dx + dy*dy

                    weight = 1.0 - r2/h2
                    if weight > 0.0:
                        tmp_variable[ix, iy, threadnum] = tmp_variable[ix, iy, threadnum] + weight*variable[ip]/norm

    # Add up contributions from each thread
    for threadnum in range(numthreads):
        for ix in range(nx):
            for iy in range(ny):
                projection[ix, iy] = projection[ix, iy] + tmp_variable[ix, iy, threadnum]

    # Fix to avoid returning a memory-view
    tmp = np.zeros((nx, ny), dtype=np.float64)
    tmp[:, :] = projection[:, :]
    return tmp

cpdef double simple_reduction(int n, int num_threads):
    cdef int i
    cdef int sum = 0


    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        sum += 1
    return sum

def get_curvature(real_t[:, :] Bvec, real_t[:, :] Bgradient):

    """
    This function returns the curvature, equivalent
    to Christoph's code:

    Kvec  = np.dot(Bgrad_mat[i,],Bvec[i,]) / B[i]**2
    Kvec -= np.dot(np.dot(Bgrad_mat[i,],Bvec[i,]),Bvec[i,]) * Bvec[i,] / B[i]**4
    K2 = (Kvec**2).sum())
    K = np.sqrt(K2)
    """
    cdef int ip, ii, jj, kk
    cdef int Np = Bvec.shape[0]
    cdef real_t B2, gradB_ij, gradB_kj, K_kk, term1, term2

    cdef real_t[:] K2 = np.zeros(Np, dtype=np.float64)

    # Bgradient has not been reshaped, \nabla B_ij
    for ip in prange(Np, nogil=True, schedule='static'):#, num_threads=32):
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
