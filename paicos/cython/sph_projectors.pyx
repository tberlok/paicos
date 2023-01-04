from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free
cimport openmp
import cython
import numpy as np
cimport numpy as np

ctypedef fused real_t:
    float
    double
    np.float32_t
    np.float64_t

STUFF = "Hi" # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu


def project_image2(real_t[:] xvec, real_t[:] yvec, real_t[:] zvec,
                   real_t[:] variable,
                   real_t[:] hvec, int nx, real_t xc, real_t yc, real_t zc,
                   real_t sidelength_x, real_t sidelength_y, real_t sidelength_z,
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
    cdef real_t x, y, z, norm
    cdef real_t x0, y0, z0
    cdef real_t boundary_factor

    assert sidelength_x/nx == sidelength_y/ny

    # Lower left corner of image in Arepo coordinates
    x0 = xc - sidelength_x/2.0
    y0 = yc - sidelength_y/2.0
    z0 = zc - sidelength_z/2.0

    for ip in range(Np):
        # Center coordinate system at x0, y0
        x = xvec[ip] - x0
        y = yvec[ip] - y0
        z = zvec[ip] - z0

        # Apply periodic boundary condition
        # if x < 0.0:
        #     x = x + boxsize
        # elif x > boxsize:
        #     x = x - boxsize

        # if y < 0.0:
        #     y = y + boxsize
        # elif y > boxsize:
        #     y = y - boxsize


        boundary_factor = 1.0

        if z < hvec[ip]:
                boundary_factor = 0.5  + z/(2.0*hvec[ip])
        if z > (sidelength_z - hvec[ip]):
                boundary_factor = 0.5 - (z - sidelength_z)/(2.0*hvec[ip])

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
                weight *= boundary_factor
                if weight > 0.0:
                    projection[ix, iy] += weight*variable[ip]/norm

    # Fix to avoid returning a memory-view
    tmp = np.zeros((nx, ny), dtype=np.float64)
    tmp[:, :] = projection[:, :]
    return tmp


def project_image2_omp(real_t[:] xvec, real_t[:] yvec, real_t[:] zvec,
                   real_t[:] variable,
                   real_t[:] hvec, int nx, real_t xc, real_t yc, real_t zc,
                   real_t sidelength_x, real_t sidelength_y, real_t sidelength_z,
                   real_t boxsize, int numthreads=1):

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
    cdef real_t x, y, z, norm
    cdef real_t x0, y0, z0
    cdef real_t boundary_factor

    assert sidelength_x/nx == sidelength_y/ny

    # Lower left corner of image in Arepo coordinates
    x0 = xc - sidelength_x/2.0
    y0 = yc - sidelength_y/2.0
    z0 = zc - sidelength_z/2.0

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
            z = zvec[ip] - z0

            # Apply periodic boundary condition
            # if x < 0.0:
            #     x = x + boxsize
            # elif x > boxsize:
            #     x = x - boxsize

            # if y < 0.0:
            #     y = y + boxsize
            # elif y > boxsize:
            #     y = y - boxsize


            boundary_factor = 1.0

            if z < hvec[ip]:
                    boundary_factor = 0.5  + z/(2.0*hvec[ip])
            if z > (sidelength_z - hvec[ip]):
                    boundary_factor = 0.5 - (z - sidelength_z)/(2.0*hvec[ip])

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
                    weight = weight * boundary_factor
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
        # if x < 0.0:
        #     x = x + boxsize
        # elif x > boxsize:
        #     x = x - boxsize

        # if y < 0.0:
        #     y = y + boxsize
        # elif y > boxsize:
        #     y = y - boxsize

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
            # if x < 0.0:
            #     x = x + boxsize
            # elif x > boxsize:
            #     x = x - boxsize

            # if y < 0.0:
            #     y = y + boxsize
            # elif y > boxsize:
            #     y = y - boxsize

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
