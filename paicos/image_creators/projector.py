"""
Defines a class that creates an image of a given variable by projecting it
onto a 2D plane.
"""
import numpy as np
from .image_creator import ImageCreator
from .. import util
from .. import settings
from .. import units


class Projector(ImageCreator):
    """
    A class that allows creating an image of a given variable by projecting
    it onto a 2D plane.

    The Projector class is a subclass of the ImageCreator class.
    The Projector class creates an image of a given variable by projecting
    it onto a 2D plane.

    It takes in several parameters such as a snapshot object, center and
    widths of the region, direction of projection, and various optional
    parameters for number of pixels, smoothing length.
    """

    def __init__(self, snap, center, widths, direction,
                 npix=512, parttype=0, nvol=8, make_snap_with_selection=True):
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

        direction : str, Orientation
            Direction of the projection, e.g. 'x', 'y' or 'z'
            or an Orientation instance.

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

        # nvol is an integer that determines the smoothing length
        self.nvol = nvol

        self._observers = []

        self.make_snap_with_selection = make_snap_with_selection

        # Call selection
        self.has_do_region_selection_been_called = False
        self._do_region_selection()
        self.has_do_region_selection_been_called = True

    def _bind_to(self, callback):
        self._observers.append(callback)

    def _do_region_selection(self):

        self.do_unit_consistency_check()

        if self.has_do_region_selection_been_called:
            if self.make_snap_with_selection:
                err_msg = ("It looks like you are changing projector "
                           + "properties after the fact, i.e. changing widths "
                           + "center, orientation, resolution etc. This does not "
                           + "work with the option make_snap_with_selection, which "
                           + "you have turned on.")
                raise RuntimeError(err_msg)
        center = self.center
        widths = self.widths
        snap = self.snap
        parttype = self.parttype

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

        # Reduce the snapshot to only contain region of interest
        if self.make_snap_with_selection:
            self.snap = self.snap.select(self.index, parttype=parttype)

        # Calculate the smoothing length
        avail_list = (list(snap.keys()) + snap._auto_list)
        if f'{parttype}_Volume' in avail_list:
            self.hsml = np.cbrt(self.nvol * (self.snap[f"{parttype}_Volume"])
                                / (4.0 * np.pi / 3.0))
        elif f'{parttype}_SubfindHsml' in avail_list:
            self.hsml = np.copy(self.snap[f'{parttype}_SubfindHsml'])
        else:
            raise RuntimeError(
                'There is no smoothing length or volume for the projector')

        self.pos = np.copy(self.snap[f'{self.parttype}_Coordinates'])

        if settings.use_units:
            self.hsml = self.hsml.to(self.pos.unit)

        if not self.make_snap_with_selection:
            self.hsml = self.hsml[self.index]
            self.pos = self.pos[self.index]

        # Call other functions that need to be updated
        for callback in self._observers:
            # print(callback, 'from projector')
            callback()

    @util.remove_astro_units
    def _cython_project(self, center, widths, variable):
        """
        Private method for projecting using cython
        """
        if settings.openMP_has_issues:
            from ..cython.sph_projectors import project_image as project
            from ..cython.sph_projectors import project_oriented_image as project_orie
        else:
            from ..cython.sph_projectors import project_image_omp as project
            from ..cython.sph_projectors import project_oriented_image_omp as project_orie

        x_c, y_c, z_c = center[0], center[1], center[2]
        width_x, width_y, width_z = widths

        boxsize = self.snap.box
        if self.direction == 'x':
            projection = project(self.pos[:, 1],
                                 self.pos[:, 2],
                                 variable,
                                 self.hsml, self.npix,
                                 y_c, z_c, width_y, width_z,
                                 boxsize, settings.numthreads_reduction)
        elif self.direction == 'y':
            projection = project(self.pos[:, 2],
                                 self.pos[:, 0],
                                 variable,
                                 self.hsml, self.npix,
                                 z_c, x_c, width_z, width_x,
                                 boxsize, settings.numthreads_reduction)
        elif self.direction == 'z':
            projection = project(self.pos[:, 0],
                                 self.pos[:, 1],
                                 variable,
                                 self.hsml, self.npix,
                                 x_c, y_c, width_x, width_y,
                                 boxsize, settings.numthreads_reduction)
        elif self.direction == 'orientation':
            unit_vectors = self.orientation.cartesian_unit_vectors

            projection = project_orie(self.pos[:, 0],
                                      self.pos[:, 1],
                                      self.pos[:, 2],
                                      variable,
                                      self.hsml, self.npix,
                                      x_c, y_c, z_c, width_x, width_y,
                                      boxsize,
                                      unit_vectors['x'],
                                      unit_vectors['y'],
                                      unit_vectors['z'],
                                      settings.numthreads_reduction)

        return projection

    def project_variable(self, variable):
        """
        projects a given variable onto a 2D plane.

        Parameters
        ----------
        variable : str, array
            The variable to be projected, it can be passed as string
            or a 1d array.

        Returns
        -------
        numpy array
            The image (2d array) of the projected variable.
        """
        self.do_unit_consistency_check()
        # This calls _do_region_selection if resolution, Orientation,
        # widths or center changed
        self._check_if_properties_changed()

        if isinstance(variable, str):
            err_msg = 'projector uses a different parttype'
            assert int(variable[0]) == self.parttype, err_msg
            variable = self.snap[variable]
        else:
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        if variable.shape == self.index.shape:
            variable = variable[self.index]

        assert len(variable.shape) == 1, 'only scalars can be projected'

        # Do the projection
        projection = self._cython_project(self.center, self.widths, variable)

        # Transpose
        projection = projection.T

        assert projection.shape[1] == self.npix_width, (projection.shape,
                                                        self.npix_width)

        assert projection.shape[0] == self.npix_height, (projection.shape,
                                                         self.npix_height)

        area_per_pixel = self.area / np.prod(projection.shape)

        if isinstance(variable, units.PaicosQuantity):
            projection = projection * variable.unit_quantity

        return projection / area_per_pixel
