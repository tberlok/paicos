import numpy as np
from . import ImageCreator
from . import util


class Projector(ImageCreator):
    """
    A class that allows creating an image of a given variable by projecting
    it onto a 2D plane.

    The Projector class is a subclass of the ImageCreator class.
    The Projector class creates an image of a given variable by projecting
    it onto a 2D plane.

    It takes in several parameters such as a snapshot object, center and
    widths of the region, direction of projection, and various optional
    parameters for number of pixels, smoothing length and number of threads
    for parallelization. It then calls various functions from the paicos
    package, including get_index_of_region, load_data, and get_volumes. This
    class also has a function called _check_if_omp_has_issues which checks if
    the parallelization via OpenMP works, and sets the number of threads
    accordingly.
    """

    def __init__(self, snap, center, widths, direction,
                 npix=512, nvol=8, make_snap_with_selection=True):

        """
        Initialize the Projector class.

        Parameters
        ----------
        snap : Snapshot
            A snapshot object of Snapshot class from paicos package.

        center : numpy array
            Center of the region on which projection is to be done, e.g.
            center = [xc, yc, zc].

        widths : numpy array
            Widths of the region on which projection is to be done,
            e.g.m widths=[width_x, width_y, width_z].

        direction : str
            Direction of the projection, e.g. 'x', 'y' or 'z'.

        npix : int, optional
            Number of pixels in the horizontal direction of the image,
            by default 512.

        nvol : int, optional
            Integer used to determine the smoothing length, by default 8

        """

        # call the superclass constructor to initialize the ImageCreator class
        super().__init__(snap, center, widths, direction, npix=npix)

        # nvol is an integer that determines the smoothing length
        self.nvol = nvol

        # check if OpenMP has any issues with the number of threads
        if util.check_if_omp_has_issues():
            self.use_omp = True
            self.numthreads = util.numthreads
        else:
            self.use_omp = False
            self.numthreads = 1

        # get the index of the region of projection
        self.index = util.get_index_of_region(self.snap["0_Coordinates"],
                                              center, widths, snap.box)

        # Reduce the snapshot to only contain region of interest
        if make_snap_with_selection:
            self.snap = self.snap.select(self.index)

        # Calculate the smoothing length
        self.hsml = np.cbrt(nvol*(self.snap["0_Volumes"]) / (4.0*np.pi/3.0))

        self.pos = self.snap['0_Coordinates']

        if not make_snap_with_selection:
            self.hsml = self.hsml[self.index]
            self.pos = self.pos[self.index]

    @util.remove_astro_units
    def _cython_project(self, center, widths, variable):
        if self.use_omp:
            from .cython.sph_projectors import project_image_omp
            project_image = project_image_omp
        else:
            from .cython.sph_projectors import project_image

        xc, yc, zc = center[0], center[1], center[2]
        width_x, width_y, width_z = widths

        boxsize = self.snap.box
        if self.direction == 'x':
            projection = project_image(self.pos[:, 1],
                                       self.pos[:, 2],
                                       variable,
                                       self.hsml, self.npix,
                                       yc, zc, width_y, width_z,
                                       boxsize, self.numthreads)
        elif self.direction == 'y':
            projection = project_image(self.pos[:, 0],
                                       self.pos[:, 2],
                                       variable,
                                       self.hsml, self.npix,
                                       xc, zc, width_x, width_z,
                                       boxsize, self.numthreads)
        elif self.direction == 'z':
            projection = project_image(self.pos[:, 0],
                                       self.pos[:, 1],
                                       variable,
                                       self.hsml, self.npix,
                                       xc, yc, width_x, width_y,
                                       boxsize, self.numthreads)

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

        from . import units

        if isinstance(variable, str):
            variable = self.snap[variable]
        else:
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        if variable.shape == self.index.shape:
            variable = variable[self.index]

        # Do the projection
        projection = self._cython_project(self.center, self.widths, variable)

        # Transpose
        projection = projection.T
        area_per_pixel = self.area/np.product(projection.shape)

        if isinstance(variable, units.PaicosQuantity):
            projection = projection*variable.unit_quantity

        return projection/area_per_pixel
