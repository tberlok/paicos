from cython.parallel import prange, parallel
cimport openmp
import numpy as np
cimport numpy as np

ctypedef fused real_t:
    float
    double
    np.float32_t
    np.float64_t

STUFF = "Hi"  # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu


def project_image(real_t[:] xvec, real_t[:] yvec, real_t[:] variable,
                  real_t[:] hvec, int nx, real_t xc, real_t yc,
                  real_t sidelength_x, real_t sidelength_y,
                  real_t boxsize, int numthreads=1):

    """

    Projects particles using an SPH-like kernel.

    This function assumes that only particles with coordinates

    x0 < x < x0 + sidelength_x
    y0 < y < y0 + sidelength_y

    are passed to it. For speed, this is not checked!

    Here x0, y0 are the coordinates at the lower, left corner.

    Parameters:
        xvec (array, N): positions along x (horizontal)
        yvec (array, N): positions along y (vertical)
        variable (array, N): variable to be projected (e.g. mass)
        hsml (array, N): size of particles
        sidelength_x (double): size of image along x
        sidelength_y (double): size of image along y
        boxsize (double): size of simulation domain,
                           which for now is assumed to be cubic!

    Returns:
        2d array: A 2D array with the projected variable.

    """

    assert numthreads == 1, 'use project_image_omp for more than one thread'

    # Number of particles
    cdef int Np = xvec.shape[0]

    # Shape of projection array
    cdef int ny = <int> round(sidelength_y/sidelength_x * nx)
    msg = '(sidelength_y/sidelength_x * nx) needs to be an integer'

    assert abs((sidelength_y/sidelength_x * nx) - <float> ny) < 1e-8, msg

    # Create projection array
    cdef real_t[:, :] projection = np.zeros((nx, ny), dtype=np.float64)

    # Loop integers and other variables
    cdef int ip, ix, iy, ih
    cdef int ipx, ipy
    cdef int ix_min, ix_max
    cdef int iy_min, iy_max
    cdef real_t dx, dy, r2, h, h2, weight
    cdef real_t x, y, norm
    cdef real_t x0, y0

    # assert sidelength_x/nx == sidelength_y/ny

    # Lower left corner of image in Arepo coordinates
    x0 = xc - sidelength_x / 2.0
    y0 = yc - sidelength_y / 2.0

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

    """
    Same as project_image but here with an openmp parallel implementation.
    """

    # Number of particles
    cdef int Np = xvec.shape[0]

    # Shape of projection array
    cdef int ny = <int> round(sidelength_y/sidelength_x * nx)
    msg = '(sidelength_y/sidelength_x * nx) needs to be an integer'

    assert abs((sidelength_y/sidelength_x * nx) - <float> ny) < 1e-8, msg

    # Loop integers and other variables
    cdef int ip, ix, iy, ih, threadnum
    cdef int ipx, ipy
    cdef int ix_min, ix_max
    cdef int iy_min, iy_max
    cdef real_t dx, dy, r2, h, h2, weight
    cdef real_t x, y, norm
    cdef real_t x0, y0

    # assert sidelength_x/nx == sidelength_y/ny

    # Lower left corner of image in Arepo coordinates
    x0 = xc - sidelength_x/2.0
    y0 = yc - sidelength_y/2.0

    cdef real_t[:, :, :] tmp_var = np.zeros((nx, ny, numthreads), dtype=np.float64)
    cdef real_t[:, :] projection = np.zeros((nx, ny), dtype=np.float64)

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
                        tmp_var[ix, iy, threadnum] = tmp_var[ix, iy, threadnum] \
                                                     + weight*variable[ip]/norm

    # Add up contributions from each thread
    for threadnum in range(numthreads):
        for ix in range(nx):
            for iy in range(ny):
                projection[ix, iy] = projection[ix, iy] + tmp_var[ix, iy, threadnum]

    # Fix to avoid returning a memory-view
    tmp = np.zeros((nx, ny), dtype=np.float64)
    tmp[:, :] = projection[:, :]
    return tmp

def project_oriented_image(real_t[:] xvec, real_t[:] yvec, real_t[:] zvec, real_t[:] variable,
                           real_t[:] hvec, int nx,
                           real_t xc, real_t yc, real_t zc,
                           real_t sidelength_x, real_t sidelength_y,
                           real_t boxsize,
                           real_t[:] unit_vector_x,
                           real_t[:] unit_vector_y,
                           real_t[:] unit_vector_z,
                           int numthreads=1):

    """
    Same as project_image but here for an arbitrarily oriented image.
    """

    assert numthreads == 1, 'use project_image_omp for more than one thread'

    # Number of particles
    cdef int Np = xvec.shape[0]

    # Shape of projection array
    cdef int ny = <int> round(sidelength_y/sidelength_x * nx)
    msg = '(sidelength_y/sidelength_x * nx) needs to be an integer'

    assert abs((sidelength_y/sidelength_x * nx) - <float> ny) < 1e-8, msg

    # Create projection array
    cdef real_t[:, :] projection = np.zeros((nx, ny), dtype=np.float64)

    # Loop integers and other variables
    cdef int ip, ix, iy, ih
    cdef int ipx, ipy
    cdef int ix_min, ix_max
    cdef int iy_min, iy_max
    cdef real_t dx, dy, r2, h, h2, weight
    cdef real_t x, y, norm
    cdef real_t cen_x, cen_y, cen_z

    # assert sidelength_x/nx == sidelength_y/ny

    for ip in range(Np):
        # Centered coordinates
        cen_x = xvec[ip] - xc
        cen_y = yvec[ip] - yc
        cen_z = zvec[ip] - zc

        # Apply boundary conditions here?

        # Projection of coordinate along perp_vector1
        x = cen_x * unit_vector_x[0] + cen_y * unit_vector_x[1] + cen_z * unit_vector_x[2]
        # Projection of coordinate along perp_vector2
        y = cen_x * unit_vector_y[0] + cen_y * unit_vector_y[1] + cen_z * unit_vector_y[2]

        # Transform these so that they are in the range [0, sidelength]
        x = x + sidelength_x/2.0
        y = y + sidelength_y/2.0

        #----------------------------------------------------------------------------------#
        # Below this line the code is identical to project_image function. Could clean this up.
        #----------------------------------------------------------------------------------#

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


def project_oriented_image_omp(real_t[:] xvec, real_t[:] yvec, real_t[:] zvec, real_t[:] variable,
                               real_t[:] hvec, int nx,
                               real_t xc, real_t yc, real_t zc,
                               real_t sidelength_x, real_t sidelength_y,
                               real_t boxsize,
                               real_t[:] unit_vector_x,
                               real_t[:] unit_vector_y,
                               real_t[:] unit_vector_z,
                               int numthreads=1):

    """
    Same as project_oriented_image but here with an openmp implementation.
    """

    # Number of particles
    cdef int Np = xvec.shape[0]

    # Shape of projection array
    cdef int ny = <int> round(sidelength_y/sidelength_x * nx)
    msg = '(sidelength_y/sidelength_x * nx) needs to be an integer'

    assert abs((sidelength_y/sidelength_x * nx) - <float> ny) < 1e-8, msg

    # Loop integers and other variables
    cdef int ip, ix, iy, ih, threadnum
    cdef int ipx, ipy
    cdef int ix_min, ix_max
    cdef int iy_min, iy_max
    cdef real_t dx, dy, r2, h, h2, weight
    cdef real_t x, y, norm
    cdef real_t cen_x, cen_y, cen_z

    #assert sidelength_x/nx == sidelength_y/ny

    # Create projection arrays
    cdef real_t[:, :, :] tmp_var = np.zeros((nx, ny, numthreads), dtype=np.float64)
    cdef real_t[:, :] projection = np.zeros((nx, ny), dtype=np.float64)

    with nogil, parallel(num_threads=numthreads):
        for ip in prange(Np, schedule='static'):
            threadnum = openmp.omp_get_thread_num()
            # Centered coordinates
            cen_x = xvec[ip] - xc
            cen_y = yvec[ip] - yc
            cen_z = zvec[ip] - zc

            # Apply boundary conditions here?

            # Projection of coordinate along perp_vector1
            x = cen_x * unit_vector_x[0] + cen_y * unit_vector_x[1] + cen_z * unit_vector_x[2]
            # Projection of coordinate along perp_vector2
            y = cen_x * unit_vector_y[0] + cen_y * unit_vector_y[1] + cen_z * unit_vector_y[2]

            # Transform these so that they are in the range [0, sidelength]
            x = x + sidelength_x/2.0
            y = y + sidelength_y/2.0

            #----------------------------------------------------------------------------------#
            # Below this line the code is identical to project_image_omp function. Could clean this up.
            #----------------------------------------------------------------------------------#

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
                        tmp_var[ix, iy, threadnum] = tmp_var[ix, iy, threadnum] \
                                                     + weight*variable[ip]/norm

    # Add up contributions from each thread
    for threadnum in range(numthreads):
        for ix in range(nx):
            for iy in range(ny):
                projection[ix, iy] = projection[ix, iy] + tmp_var[ix, iy, threadnum]

    # Fix to avoid returning a memory-view
    tmp = np.zeros((nx, ny), dtype=np.float64)
    tmp[:, :] = projection[:, :]
    return tmp
