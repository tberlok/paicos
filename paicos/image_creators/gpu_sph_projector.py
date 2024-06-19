import numpy as np
from .image_creator import ImageCreator
from .. import util
from .. import settings
from .. import units

import cupy as cp
from numba import cuda


@cuda.jit(device=True, inline=True)
def get_norm_and_index(xp, yp, hp, extra):
    ipx = int(xp)
    ipy = int(yp)

    h2 = hp**2

    norm = 0.0
    for ix in range(-extra, extra):
        for iy in range(-extra, extra):
            dx = xp - 0.5 - ix - ipx
            dy = yp - 0.5 - iy - ipy
            r2 = dx * dx + dy * dy

            weight = 1.0 - r2 / h2

            if weight < 0.0:
                weight = 0.0

            norm += weight

    return norm, ipx, ipy


@cuda.jit(device=True, inline=True)
def get_weight(ix, iy, xp, yp, hp, varp):
    ipx = int(xp)
    ipy = int(yp)

    h2 = hp**2

    dx = xp - 0.5 - ix - ipx
    dy = yp - 0.5 - iy - ipy
    r2 = dx * dx + dy * dy

    weight = 1.0 - r2 / h2
    # return weight
    if weight < 0.0:
        return 0.0
    else:
        return weight * varp


@cuda.jit(max_registers=64)
def deposit(nx, ny, x, y, z, hsml, variable, image1, image2, image4, image8, image16,
            widths, center, unit_vector_x, unit_vector_y, unit_vector_z, extra):
    # threadindex
    ip = cuda.grid(1)

    # particle properties
    hp = hsml[ip]
    xp = x[ip]
    yp = y[ip]
    zp = z[ip]
    varp = variable[ip]

    # Center of projection
    xc, yc, zc = center

    # Centered coordinates
    cen_x = xp - xc
    cen_y = yp - yc
    cen_z = zp - zc

    # Do projections along unit vectors
    # Projection of coordinate along perp_vector1
    xp = cen_x * unit_vector_x[0] + cen_y * \
        unit_vector_x[1] + cen_z * unit_vector_x[2]
    # Projection of coordinate along perp_vector2
    yp = cen_x * unit_vector_y[0] + cen_y * \
        unit_vector_y[1] + cen_z * unit_vector_y[2]
    # Projection of coordinate along normal_vector
    zp = cen_x * unit_vector_z[0] + cen_y * \
        unit_vector_z[1] + cen_z * unit_vector_z[2]

    # For oriented images the projection is rotated to be
    # in a coordinate system where the projection direction is
    # z. The sidelength are the x-axis and y-axis of the image,
    # not the x and y dimensions of the simulation
    sidelength_x, sidelength_y, sidelength_z = widths

    # Check if this cell/particle is inside domain
    inside_domain = False
    if (xp < sidelength_x / 2.0) and (xp > -sidelength_x / 2.0):
        if (yp < sidelength_y / 2.0) and (yp > -sidelength_y / 2.0):
            if (zp < sidelength_z / 2.0):
                if (zp > -sidelength_z / 2.0):
                    inside_domain = True

    if inside_domain:

        # Transform these so that they are in the range [0, sidelength]
        xp = xp + sidelength_x / 2.0
        yp = yp + sidelength_y / 2.0

        # Image cell size on finest grid
        dx = sidelength_x / nx

        # Position of particle in units of sidelength (0, sidelength)
        xp = xp / dx
        yp = yp / dx
        hp = hp / dx

        # Use a minimum cell size for projection
        if hp < 1.0:
            hp = 1.0

        # Split into the 5 grids at different resolutions
        if hp < extra:
            im = 1
        elif hp < 2 * extra:
            im = 2
        elif hp < 4 * extra:
            im = 4
        elif hp < 8 * extra:
            im = 8
        else:
            im = 16

        # Change cell size depending on the image cell size
        xp /= im
        yp /= im
        hp /= im

        # Find norm of particle (ensures that all its 'mass' is deposited)
        norm, ipx, ipy = get_norm_and_index(xp, yp, hp, extra)

        # Loop over the stencil
        for d_ix in range(-extra, extra):
            for d_iy in range(-extra, extra):
                ix = ipx + d_ix
                iy = ipy + d_iy
                if ix >= 0 and ix < nx // im and iy >= 0 and iy < ny // im:
                    weight = get_weight(d_ix, d_iy, xp, yp, hp, varp)
                    if im == 1:
                        cuda.atomic.add(image1, (ix, iy), weight / norm)
                    elif im == 2:
                        cuda.atomic.add(image2, (ix, iy), weight / norm)
                    elif im == 4:
                        cuda.atomic.add(image4, (ix, iy), weight / norm)
                    elif im == 8:
                        cuda.atomic.add(image8, (ix, iy), weight / norm)
                    elif im == 16:
                        cuda.atomic.add(image16, (ix, iy), weight / norm)


class GpuSphProjector(ImageCreator):
    """
    A class that allows creating an image of a given variable by projecting
    it onto a 2D plane.

    This GPU implementation of SPH-like projection splits the particles
    to be deposited into five nested grids. The current implementation
    gives visual artifacts when the number of pixels is chosen to be too
    large compared to the gas cell sizes (i.e. when trying
    to make a high-resolution image of a low resolution simulation).
    This projector class is therefore for the moment only recommended
    for interactive exploration or for creating movies.

    The input pixel size needs to be a power of 2 in the horizontal direction.
    """

    def __init__(self, snap, center, widths, direction,
                 npix=512, parttype=0, nvol=8, threadsperblock=8,
                 do_pre_selection=False):
        """
        Initialize the Projector class.

        Parameters
        ----------
        snap : Snapshot
            A snapshot object of Snapshot class from paicos package.

        center : numpy array
            Center of the region on which projection is to be done, e.g.
            center = [x_c, y_c, z_c].

        widths : numpy array
            Widths of the region on which projection is to be done,
            e.g.m widths=[width_x, width_y, width_z].

        direction : str
            Direction of the projection, e.g. 'x', 'y' or 'z'.

        npix : int, optional
            Number of pixels in the horizontal direction of the image,
            by default 512.

        parttype : int, optional
            Number of the particle type to project, by default gas (PartType 0).

        nvol : int, optional
            Integer used to determine the smoothing length, by default 8

        """

        # call the superclass constructor to initialize the ImageCreator class
        super().__init__(snap, center, widths, direction, npix=npix, parttype=parttype)
        err_msg = 'Pixel sizes in both directions must be divisible by 32'
        assert self.npix_width % 32 == 0, err_msg
        assert self.npix_height % 32 == 0, err_msg

        parttype = self.parttype

        self.threadsperblock = threadsperblock

        self.do_pre_selection = do_pre_selection

        # nvol is an integer that determines the smoothing length
        self.nvol = nvol

        # Calculate the smoothing length
        avail_list = (list(snap.keys()) + snap._auto_list)
        if f'{parttype}_Volume' in avail_list:
            self.hsml = np.cbrt(nvol * (self.snap[f"{parttype}_Volume"])
                                / (4.0 * np.pi / 3.0))
        elif f'{parttype}_SubfindHsml' in avail_list:
            self.hsml = self.snap[f'{parttype}_SubfindHsml']
        else:
            raise RuntimeError(
                'There is no smoothing length or volume for the projector')

        self.pos = self.snap[f'{self.parttype}_Coordinates']

        if settings.use_units:
            self.hsml = self.hsml.to(self.pos.unit)

        # TODO: add split into GPU and CPU based on cell sizes here.
        # Perhaps make the CPU do everything above grid8.

        self._sph_kernel_radius = 4

        # Call selection
        self.has_do_region_selection_been_called = False
        self._do_region_selection()
        self.has_do_region_selection_been_called = True

    def _do_region_selection(self):

        self.do_unit_consistency_check()

        center = self.center
        widths = self.widths
        snap = self.snap
        parttype = self.parttype

        if self.has_do_region_selection_been_called:
            if self.do_pre_selection:
                import warnings
                err_msg = ("It looks like you are changing projector "
                           + " properties after the fact, i.e. changing widths "
                           + "center, orientation, resolution etc. This might be "
                           + "slow with the option do_pre_selection, which "
                           + "you have turned on. If your GPU has enough memory "
                           + "then it is probably better to set do_pre_selection "
                           + "to False.")
                warnings.warn(err_msg)

        # Send subset of snapshot to GPU
        if self.do_pre_selection:
            # get the index of the region of projection
            if self.direction != 'orientation':
                get_index = util.get_index_of_cubic_region
                self.index = get_index(self.snap[f"{parttype}_Coordinates"],
                                       center, widths, snap.box)
            else:
                get_index = util.get_index_of_rotated_cubic_region
                self.index = get_index(snap[f"{parttype}_Coordinates"],
                                       center, widths, snap.box,
                                       self.orientation)

            self.hsml = self.hsml[self.index]
            self.pos = self.pos[self.index]

            self._send_data_to_gpu()
            self._send_small_data_to_gpu()
        # Send entirety of snapshot to GPU (if we have not already
        # done so). Always send small data with change in resolution etc
        else:
            if not self.has_do_region_selection_been_called:
                self._send_data_to_gpu()
            self._send_small_data_to_gpu()

    def _send_data_to_gpu(self):
        self.gpu_variables = {}
        if settings.use_units:
            self.gpu_variables['x'] = cp.array(self.pos[:, 0].value)
            self.gpu_variables['y'] = cp.array(self.pos[:, 1].value)
            self.gpu_variables['z'] = cp.array(self.pos[:, 2].value)
            self.gpu_variables['hsml'] = cp.array(self.hsml.value)
        else:
            self.gpu_variables['x'] = cp.array(self.pos[:, 0])
            self.gpu_variables['y'] = cp.array(self.pos[:, 1])
            self.gpu_variables['z'] = cp.array(self.pos[:, 2])
            self.gpu_variables['hsml'] = cp.array(self.hsml)

        self._send_small_data_to_gpu()

    def _send_small_data_to_gpu(self):
        unit_vectors = self.orientation.cartesian_unit_vectors
        self.gpu_variables['unit_vector_x'] = cp.array(unit_vectors['x'])
        self.gpu_variables['unit_vector_y'] = cp.array(unit_vectors['y'])
        self.gpu_variables['unit_vector_z'] = cp.array(unit_vectors['z'])

        if settings.use_units:
            self.gpu_variables['widths'] = cp.array(self.widths.value)
            self.gpu_variables['center'] = cp.array(self.center.value)
        else:
            self.gpu_variables['widths'] = cp.array(self.widths)
            self.gpu_variables['center'] = cp.array(self.center)

    def _increase_image_resolution(self, image, factor):
        """
        Increase the number of pixes without changing the total 'mass'
        of the image
        """
        if factor == 1:
            return image
        repeats = factor
        new_image = cp.repeat(cp.repeat(image, repeats, axis=0),
                              repeats, axis=1)
        return new_image / factor**2

    def _get_full_image_as_numpy(self, image1, image2, image4, image8, image16):
        image8 += self._increase_image_resolution(image16, 2)
        image4 += self._increase_image_resolution(image8, 2)
        image2 += self._increase_image_resolution(image4, 2)
        image1 += self._increase_image_resolution(image2, 2)
        return cp.asnumpy(image1)

    def _gpu_project(self, variable_str):
        """
        Private method for projecting using cuda code
        """

        unit_vector_x = self.gpu_variables['unit_vector_x']
        unit_vector_y = self.gpu_variables['unit_vector_y']
        unit_vector_z = self.gpu_variables['unit_vector_z']
        x = self.gpu_variables['x']
        y = self.gpu_variables['y']
        z = self.gpu_variables['z']
        widths = self.gpu_variables['widths']
        center = self.gpu_variables['center']
        hsml = self.gpu_variables['hsml']
        nx = self.npix_width
        ny = self.npix_height
        variable = self.gpu_variables[variable_str]

        threadsperblock = self.threadsperblock
        blockspergrid = (x.size + (threadsperblock - 1)) // threadsperblock
        image1 = cp.zeros((nx, ny))
        image2 = cp.zeros((nx // 2, ny // 2))
        image4 = cp.zeros((nx // 4, ny // 4))
        image8 = cp.zeros((nx // 8, ny // 8))
        image16 = cp.zeros((nx // 16, ny // 16))

        deposit[blockspergrid, threadsperblock](nx, ny, x, y, z, hsml, variable,
                                                image1, image2, image4, image8, image16,
                                                widths, center,
                                                unit_vector_x,
                                                unit_vector_y,
                                                unit_vector_z, self._sph_kernel_radius)

        projection = self._get_full_image_as_numpy(
            image1, image2, image4, image8, image16)
        return projection

    def project_variable(self, variable):
        """
        projects a given variable onto a 2D plane.

        Parameters
        ----------
        variable : str, function, numpy array
            variable, it can be passed as string or an array

        Returns
        -------
        numpy array
            The image of the projected variable
        """

        self.do_unit_consistency_check()

        # This calls _do_region_selection if resolution, Orientation,
        # widths or center changed
        self._check_if_properties_changed()

        if isinstance(variable, str):
            variable_str = str(variable)
            err_msg = 'projector uses a different parttype'
            assert int(variable[0]) == self.parttype, err_msg
            variable = self.snap[variable]
        else:
            variable_str = 'projection_variable'
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        assert len(variable.shape) == 1, 'only scalars can be projected'

        # Select same part of array that the projector has selected
        if self.do_pre_selection:
            variable = variable[self.index]

        if variable_str in self.gpu_variables and variable_str != 'projection_variable':
            pass
        else:
            # Send variable to gpu
            if settings.use_units:
                self.gpu_variables[variable_str] = cp.array(variable.value)
            else:
                self.gpu_variables[variable_str] = cp.array(variable)

        # Do the projection
        projection = self._gpu_project(variable_str)

        # Transpose
        projection = projection.T

        assert projection.shape[0] == self.npix_height
        assert projection.shape[1] == self.npix_width

        if isinstance(variable, units.PaicosQuantity):
            projection = projection * variable.unit_quantity

        return projection / self.area_per_pixel

    def __del__(self):
        """
        Clean up like this? Not sure it is needed...
        """
        self.release_gpu_memory()

    def release_gpu_memory(self):
        if hasattr(self, 'gpu_variables'):
            for key in list(self.gpu_variables):
                del self.gpu_variables[key]
            del self.gpu_variables

        cp._default_memory_pool.free_all_blocks()
