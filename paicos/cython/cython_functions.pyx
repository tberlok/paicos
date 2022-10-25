from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free
cimport openmp
import cython
import numpy as np
ctypedef float real_t

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
    tmp = np.zeros(Np, dtype=np.bool)
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
                  real_t boxsize):
    """

    xc, yc are the coordinates at the center of a square image
    with sidelength=sidelength.
    This function assumes that only particles with coordinates

    x0 < x < x0 + sidelength_x
    y0 < y < y0 + sidelength_y

    are passed to it. For speed, this is not checked!

    Here x0, y0 are the coordinates at the lower, left corner.
    """

    # Number of particles
    cdef int Np = xvec.shape[0]

    # Shape of projection array
    cdef int ny = <int> (sidelength_y/sidelength_x * nx)
    assert ny > 64, 'rounding error above can give wrong aspect ratio.'

    # Create projection array
    cdef real_t[:,:] projection = np.zeros((nx, ny), dtype=np.float32)

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
    x0 = xc - sidelength_x/2
    y0 = yc - sidelength_y/2

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
                dx = x - <real_t> ix
                dy = y - <real_t> iy
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
                dx = x - <real_t> ix
                dy = y - <real_t> iy
                r2 = dx*dx + dy*dy

                weight = 1.0 - r2/h2
                if weight > 0.0:
                    projection[ix, iy] += weight*variable[ip]/norm

    # Fix to avoid returning a memory-view
    tmp = np.zeros((nx, ny), dtype=np.float32)
    tmp[:, :] = projection[:, :]
    return tmp

def deposit2D(real_t[:] xvec, real_t[:] yvec, real_t[:] variable, real_t[:] hvec,
            real_t[:, :] projection, real_t boxsize):

    # Number of particles
    cdef int Np = xvec.shape[0]

    # Shape of projection array
    cdef int nx = projection.shape[0]
    cdef int ny = projection.shape[1]

    # Loop integers
    cdef int ip, ix, iy, ih
    cdef int ipx, ipy
    cdef int ix_min, ix_max
    cdef int iy_min, iy_max
    cdef real_t tx, ty, dx, dy, r2, h, h2, weight
    cdef real_t x, y, norm

    # Density deposition
    for ip in range(Np):
        # Position of particle in units of boxsize (0, boxsize)
        x = xvec[ip]*nx/boxsize
        y = yvec[ip]*ny/boxsize
        h = hvec[ip]*ny/boxsize

        # Index of closest grid point
        ipx = <int> x
        ipy = <int> y

        # Smoothing length as integer
        ih = <int> h + 1

        # Square of smoothing length
        h2 = h*h

        if h2 < 1.0:
            h2 = 1.0

        # Find minimum and maximum integers
        ix_min = ipx - ih
        iy_min = ipy - ih
        ix_max = ipx + ih
        iy_max = ipy + ih

        norm = 0.0
        for ix in range(ix_min, ix_max):
            for iy in range(iy_min, iy_max):
                dx = x - <real_t> ix
                dy = y - <real_t> iy
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
                dx = x - <real_t> ix
                dy = y - <real_t> iy
                r2 = dx*dx + dy*dy

                weight = 1.0 - r2/h2
                if weight > 0.0:
                    projection[ix, iy] += weight*variable[ip]/norm

# Seems a problem on Darwin has appeared... Some of my old code now has this
# problem with reduction operators...
# https://stackoverflow.com/questions/54776301/cython-prange-is-repeating-not-parallelizing
def deposit2D_omp(real_t[:] xvec, real_t[:] yvec, real_t[:] variable, real_t[:] hvec,
            real_t[:, :] projection, real_t boxsize):

    # Number of particles
    cdef int Np = xvec.shape[0]

    # Shape of projection array
    cdef int nx = projection.shape[0]
    cdef int ny = projection.shape[1]
    cdef int size = nx*ny

    # Loop integers
    cdef int ip, ix, iy, ih, k
    cdef int ipx, ipy
    cdef int ix_min, ix_max
    cdef int iy_min, iy_max
    cdef real_t tx, ty, dx, dy, r2, h, h2, weight
    cdef real_t x, y, norm

    cdef int threadnum, maxthreads

    maxthreads = openmp.omp_get_max_threads()

    cdef real_t[:,:, :] tmp_variable = np.zeros((nx, ny, maxthreads), dtype=np.float32)

    with nogil, parallel():
        for ip in prange(Np, schedule='static'):
            threadnum = openmp.omp_get_thread_num()
            # Position of particle in units of boxsize (0, boxsize)
            x = xvec[ip]*nx/boxsize
            y = yvec[ip]*ny/boxsize
            h = hvec[ip]*ny/boxsize

            # Index of closest grid point
            ipx = <int> x
            ipy = <int> y

            # Smoothing length as integer
            ih = <int> h + 1

            # Square of smoothing length
            h2 = h*h

            if h2 < 1.0:
                h2 = 1.0

            # Find minimum and maximum integers
            ix_min = ipx - ih
            iy_min = ipy - ih
            ix_max = ipx + ih
            iy_max = ipy + ih

            norm = 0.0
            for ix in range(ix_min, ix_max):
                for iy in range(iy_min, iy_max):
                    dx = x - <real_t> ix
                    dy = y - <real_t> iy
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
                    dx = x - <real_t> ix
                    dy = y - <real_t> iy
                    r2 = dx*dx + dy*dy

                    weight = 1.0 - r2/h2
                    if weight > 0.0:
                        tmp_variable[ix, iy, threadnum] += weight*variable[ip]/norm

    # Add up contributions from each thread
    for threadnum in range(maxthreads):
        for ix in range(ny):
            for iy in range(nx):
                projection[ix, iy] += tmp_variable[ix, iy, threadnum]/maxthreads

# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

from cython.parallel import parallel, prange

cpdef double simple_reduction(int n, int num_threads):
    cdef int i
    cdef int sum = 0


    for i in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        sum += 1
    return sum
