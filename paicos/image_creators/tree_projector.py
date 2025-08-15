"""
Defines a class that creates an image of a given variable by projecting
the Voronoi cells closest to the line of sight using a KDTree.
"""
import numpy as np
from scipy.spatial import KDTree
from .image_creator import ImageCreator
from .. import util
from .. import settings


class TreeProjector(ImageCreator):
    """
    This implements projection of gas variables by adding up several slices.
    """

    @util.conditional_timer
    def __init__(self, snap, center, widths, direction,
                 npix=512, npix_depth=None, parttype=0, make_snap_with_selection=False,
                 tol=1, verbose=False, timing=False):

        """
        Initialize the Slicer object.

        Parameters
        ----------
        snap : Snapshot
            A snapshot object of Snapshot class from paicos package.

        center :
            Center of the region on which projection is to be done, e.g.
            center = [x_c, y_c, z_c].

        widths :
            Widths of the region on which projection is to be done,
            e.g. widths=[width_x, width_y, width_z].

        direction : str, Orientation
            Direction of the projection, e.g. 'x', 'y' or 'z',
            or an Orientation instance.

        npix : int, optional
            Number of pixels in the horizontal direction of the image,
            by default 512.

        parttype : int, optional
            The particle type to project, by default gas (PartType 0).

        npix_depth: int, optional
            Number of pixels in the depth direction, by default set
            automatically based on the smallest cell sizes in the region
            and the tolerance parameter, tol (see below).

        tol: float, optional
            Smaller values of tol adds more slices to the integration.
            Convergence is expected for tol ≤ 1 but you can experiment with
            higher values.

        make_snap_with_selection : bool
            a boolean indicating if a new snapshot object should be made with
            the selected region, defaults to False.

        """

        if make_snap_with_selection:
            raise RuntimeError('make_snap_with_selection not yet implemented!')

        super().__init__(snap, center, widths, direction, npix=npix, parttype=parttype)

        self.npix_depth = npix_depth
        self.tol = tol
        self.verbose = verbose

        self._do_region_selection()

    def _do_region_selection(self):

        self.do_unit_consistency_check()

        # print('_do_region_selection was called from Slicer')
        parttype = self.parttype
        snap = self.snap
        center = self.center
        widths = self.widths

        # Pre-select a narrow region around the region-of-interest
        avail_list = (list(snap.keys()) + snap._auto_list)
        if f'{parttype}_Volume' in avail_list:
            thickness = 4.0 * np.cbrt((snap[f"{parttype}_Volume"]) / (4.0 * np.pi / 3.0))
        elif f'{parttype}_SubfindHsml' in avail_list:
            thickness = np.copy(snap[f'{parttype}_SubfindHsml'])
        else:
            err_msg = ("There is no smoothing length or volume for calculating"
                       + "the thickness of the slice")
            raise RuntimeError(err_msg)

        if hasattr(thickness, 'unit'):
            thickness = thickness.to(center.unit)

        if self.direction != 'orientation':
            get_index = util.get_index_of_cubic_region_plus_thin_layer
            self.box_selection = get_index(snap[f"{parttype}_Coordinates"],
                                           center, widths, thickness,
                                           snap.box)
        else:
            get_index = util.get_index_of_rotated_cubic_region_plus_thin_layer
            self.box_selection = get_index(snap[f"{parttype}_Coordinates"],
                                           center, widths, thickness, snap.box,
                                           self.orientation)

        if self.verbose:
            print('Sub-selection [DONE]')

        min_thickness = np.min(thickness)

        # Automatically set numbers of pixels in depth direction based on cell sizes
        if self.npix_depth is None:
            npix_depth = int(np.ceil(self.depth / min_thickness / self.tol))
            if self.verbose:
                print(f'npix_depth is {npix_depth}')

            self.npix_depth = npix_depth

        self.index_in_box_region = np.arange(snap[f"{self.parttype}_Coordinates"].shape[0]
                                             )[self.box_selection]

        # Construct a tree
        self.pos = snap[f"{parttype}_Coordinates"][self.box_selection]
        tree = KDTree(self.pos)
        if self.verbose:
            print('Tree construction [DONE]')

        # Now construct the image grid
        w, h = self._get_width_and_height_arrays()

        center = self.center
        ones = np.ones(w.shape[0])

        # Full index
        self.index = np.empty((self.npix_height, self.npix_width, self.npix_depth),
                              dtype=np.int64)

        self.delta_depth = self.depth / self.npix_depth
        depth_vector = np.arange(self.npix_depth) * self.delta_depth \
            + self.delta_depth / 2 - self.depth / 2

        for ii, dep in enumerate(depth_vector):
            if self.direction == 'x':
                image_points = np.vstack([ones * (center[0] + dep), w, h]).T
            elif self.direction == 'y':
                image_points = np.vstack([h, ones * (center[1] + dep), w]).T
            elif self.direction == 'z':
                image_points = np.vstack([w, h, ones * (center[2] + dep)]).T
            elif self.direction == 'orientation':
                orientation = self.orientation
                image_points = np.vstack([w, h, ones * (center[2] + dep)]).T - self.center
                image_points = np.matmul(orientation.rotation_matrix, image_points.T).T \
                    + self.center
            else:
                raise RuntimeError(f"Problem with direction={self.direction} input")

            # Query the tree to obtain closest Voronoi cell indices
            d, i = tree.query(image_points, workers=settings.numthreads)

            slice_index = self._unflatten(self.index_in_box_region[i])
            self.distance_to_nearest_cell = self._unflatten(d)

            if min_thickness < self.delta_depth / self.tol:
                print(f'Warning: Minimum cell size {min_thickness} is '
                      + f'less than {self.tol} of delta_depth '
                      + f'{self.delta_depth}. You should probably increase '
                      + f'npix_depth from its current value of {self.npix_depth}. '
                      + 'Image convergence is expected for npix_depth='
                      + f'{int(self.depth/min_thickness)}.')

            self.index[:, :, ii] = slice_index

    def _get_width_and_height_arrays(self):
        """
        Get width and height coordinates in the image as 1D arrays
        of total length npix_width × npix_height.
        """

        # TODO: Make this part of the image_creator class
        extent = self.extent

        npix_width = self.npix_width
        npix_height = self.npix_height
        width = self.width
        height = self.height

        w = extent[0] + (np.arange(npix_width) + 0.5) * width / npix_width
        h = extent[2] + (np.arange(npix_height) + 0.5) * height / npix_height

        if settings.use_units:
            wu = w.unit_quantity
            ww, hh = np.meshgrid(w.value, h.value)
            ww = ww * wu
            hh = hh * wu
        else:
            ww, hh = np.meshgrid(w, h)

        w = ww.flatten()
        h = hh.flatten()

        np.testing.assert_array_equal(ww, self._unflatten(ww.flatten()))

        return w, h

    def _unflatten(self, arr):
        """
        Helper function to un-flatten 1D arrays to a 2D image
        """
        return arr.flatten().reshape((self.npix_height, self.npix_width))

    @util.conditional_timer
    def project_variable(self, variable, additive=True, extrinsic=None, timing=False):
        """
        Project gas variable based on the Voronoi cells closest to each
        line of sight.

        The behavior of this method depends on whether the variable to be
        projected is additive (extrinsic) or not (intrinsic).
        (see e.g. https://en.wikipedia.org/wiki/Intensive_and_extensive_properties).

        For non-additive (intrinsic) properties, e.g. the density ρ, the returned
        projection, P, is

        P = 1/L ∫ ρ dl ,

        where L is the depth of the projection. That is, it is simply the mean
        of a number of slices.

        For additive (extrinsic) properties, e.g. mass M, the returned projection
        is instead

        P = 1 / dA ∫ dM

        where dA is the area per pixel and dM is the mass inside a voxel.

        Parameters
        ----------
        variable : str, array
            The variable to be projected, it can be passed as string
            or a 1d array.

        additive : bool
            A boolean indicating whether the variable to be
            projected is additive (e.g. Masses, Volumes)
            or not (e.g. Temperature, density).
            This parameter was previously named 'extrinsic'.

        Returns:

            projection : 2d arr
                A 2d array representing the projected gas variable.

                For non-additive (intrinsic) variables, the unit of the projection is
                identical to the unit of the input variable. For additive (extrinsic)
                variables, the projection unit is the input variable unit divided by area.

        Examples
        ----------

        We can compute the projected density in two ways, using either
        additive or non-additive projetions. In both case,
        we first need to create a TreeProjector object::

            tree_projector = pa.TreeProjector(snap, center, widths, 'z')

        Method I (divide projected mass by projected volume)::

            tree_M = tree_projector.project_variable('0_Masses', additive=True)
            tree_V = tree_projector.project_variable('0_Volume', additive=True)

            tree_rho = tree_M / tree_V

        Method II (directly project the density)::

            rho = tree_projector.project_variable('0_Density', additive=False)

        """
        # This calls _do_region_selection if resolution, Orientation,
        # widths or center changed
        self._check_if_properties_changed()

        if extrinsic is not None:
            import warnings
            warnings.warn("The keyword 'extrinsic' has been replaced by 'additive'."
                          + " The support for 'extrinsic' will be removed eventually.")
            additive = extrinsic

        parttype = self.parttype

        if isinstance(variable, str):
            assert int(variable[0]) == parttype, 'projector uses a different parttype'
            variable = self.snap[variable]
        else:
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        assert len(variable.shape) == 1, 'only scalars can be projected'

        if additive:
            avail_list = (list(self.snap.keys()) + self.snap._auto_list)
            if f'{parttype}_Volume' in avail_list:
                variable_density = variable[self.index] / self.snap[f'{parttype}_Volume'][self.index]

                projection = np.sum(variable_density * self.delta_depth, axis=2)
                """
                # Note that the above is equivalent to:
                volume_per_voxel = area_per_pixel * self.delta_depth
                projection = np.sum(variable_density * volume_per_voxel, axis=2) \
                             / area_per_pixel
                """
            else:
                err_msg = (f"The volume field for parttype {parttype} is required when"
                           + "using additive=True")
                raise RuntimeError(err_msg)
        else:
            variable = variable[self.index]
            projection = np.sum(variable * self.delta_depth, axis=2) / self.depth
            """
            # Note that the above is equivalent to:
            projection = np.mean(variable, axis=2)
            since self.delta_depth is a constant.
            """

        return projection
