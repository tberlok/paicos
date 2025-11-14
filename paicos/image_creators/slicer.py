"""
Defines a class that creates an image of a given variable by finding
the Voronoi cells closest to the image plane.
"""
import numpy as np
from scipy.spatial import KDTree
from .image_creator import ImageCreator
from .. import util
from .. import settings
from ..orientation import Orientation


class Slicer(ImageCreator):
    """
    This implements slicing of gas variables.
    """

    def __init__(self, snap, center, widths, direction,
                 npix=512, parttype=0, make_snap_with_selection=False):
        """
        Initialize the Slicer object.

        Parameters
        ----------
        snap : Snapshot
            A snapshot object of Snapshot class from paicos package.

        center :
            Center of the region on which slicing is to be done, e.g.
            center = [x_c, y_c, z_c].

        widths :
            Widths of the region on which slicing is to be done,
            e.g. widths=[width_x, width_y, width_z] where one of the widths
            is zero (e.g. width_x=0 if direction='x').

        direction : str, Orientation
            Direction of the slicing, e.g. 'x', 'y' or 'z'. For instance,
            setting direction to 'x' gives a slice in the yz plane with the
            constant x-value set to be x_c.
            For a more general orientation, one can pass an Orientation
            instance.

        npix : int, optional
            Number of pixels in the horizontal direction of the image,
            by default 512.

        parttype : int, optional
            The particle type to project, by default gas (PartType 0).

        make_snap_with_selection : bool
            a boolean indicating if a new snapshot object should be made with
            the selected region, defaults to False

        """

        if make_snap_with_selection:
            raise RuntimeError('make_snap_with_selection not yet implemented!')

        super().__init__(snap, center, widths, direction, npix=npix, parttype=parttype)

        for ii, direc in enumerate(['x', 'y', 'z']):
            if self.direction == direc:
                assert self.widths[ii] == 0.
        if self.direction == 'orientation':
            assert self.widths[2] == 0.

        self._do_region_selection()

    def _do_region_selection(self):

        self.do_unit_consistency_check()

        parttype = self.parttype
        snap = self.snap
        center = self.center
        widths = self.widths

        # Pre-select a narrow region around the region-of-interest
        avail_list = (list(snap.keys()) + snap._auto_list)
        if f'{parttype}_Volume' in avail_list:
            thickness = 4.0 * np.cbrt((snap[f"{parttype}_Volume"])
                                      / (4.0 * np.pi / 3.0))
        elif f'{parttype}_SubfindHsml' in avail_list:
            thickness = np.copy(snap[f'{parttype}_SubfindHsml'])
        else:
            raise RuntimeError(
                'There is no smoothing length or volume for the thickness of the slice')

        if hasattr(thickness, 'unit'):
            thickness = thickness.to(center.unit)

        if self.direction != 'orientation':
            get_index = util.get_index_of_cubic_region_plus_thin_layer
            self.slice = get_index(snap[f"{parttype}_Coordinates"],
                                   center, widths, thickness,
                                   snap.box)
        else:
            get_index = util.get_index_of_rotated_cubic_region_plus_thin_layer
            self.slice = get_index(snap[f"{parttype}_Coordinates"],
                                   center, widths, thickness, snap.box,
                                   self.orientation)

        self.index_in_slice_region = np.arange(snap[f"{parttype}_Coordinates"].shape[0]
                                               )[self.slice]

        # Construct a tree
        self.pos = snap[f"{parttype}_Coordinates"][self.slice]
        tree = KDTree(self.pos)

        # Now construct the image grid
        w, h = self._get_width_and_height_arrays()

        center = self.center
        ones = np.ones(w.shape[0])
        if self.direction == 'x':
            image_points = np.vstack([ones * center[0], w, h]).T
        elif self.direction == 'y':
            image_points = np.vstack([h, ones * center[1], w]).T
        elif self.direction == 'z':
            image_points = np.vstack([w, h, ones * center[2]]).T
        elif self.direction == 'orientation':
            orientation = self.orientation
            image_points = np.vstack([w, h, ones * center[2]]).T - self.center
            image_points = np.matmul(orientation.rotation_matrix, image_points.T).T \
                + self.center
        else:
            raise RuntimeError(f"Problem with direction={self.direction} input")

        # Query the tree to obtain closest Voronoi cell indices
        d, i = tree.query(image_points, workers=settings.numthreads)

        self.index = self._unflatten(self.index_in_slice_region[i])
        self.distance_to_nearest_cell = self._unflatten(d)

    def slice_variable(self, variable):
        """
        Slice a gas variable based on the Voronoi cells closest to the image
        plane.

        Parameters
        ----------
            variable: str, arr
                The variable to slice. For instance '0_Density' or snap['0_Density'].

        Returns
        -------
            slice : 2d arr
                A 2d array of the sliced variable.
        """

        # This calls _do_region_selection if resolution, Orientation,
        # widths or center changed
        self._check_if_properties_changed()

        if isinstance(variable, str):
            err_msg = 'slicer uses a different parttype'
            assert int(variable[0]) == self.parttype, err_msg
            variable = self.snap[variable]
        else:
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        return variable[self.index]

    @property
    def depth(self):
        """
        Depth of the slice, which is zero by definition.
        """
        if self.direction == 'x':
            return self.width_x

        elif self.direction == 'y':
            return self.width_y

        elif self.direction == 'z' or self.direction == 'orientation':
            return self.width_z

    @depth.setter
    def depth(self, value):
        err_msg = "You can't change the depth of a slicer, as it is zero by definition"
        raise RuntimeError(err_msg)
