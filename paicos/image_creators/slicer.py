"""
Defines a class that creates an image of a given variable by finding
the Voronoi cells closest to the image plane.
"""
import numpy as np
from scipy.spatial import KDTree
from .image_creator import ImageCreator
from .. import util
from .. import settings


class Slicer(ImageCreator):
    """
    This implements slicing of gas variables.
    """

    def __init__(self, snap, center, widths, direction,
                 npix=512, make_snap_with_selection=False):

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

        direction : str
            Direction of the slicing, e.g. 'x', 'y' or 'z'. For instance,
            setting direction to 'x' gives a slice in the yz plane with the
            constant x-value set to be x_c.

        npix : int, optional
            Number of pixels in the horizontal direction of the image,
            by default 512.

        make_snap_with_selection : bool
            a boolean indicating if a new snapshot object should be made with
            the selected region, defaults to False

        """

        if make_snap_with_selection:
            raise RuntimeError('make_snap_with_selection not yet implemented!')

        super().__init__(snap, center, widths, direction, npix=npix)

        for ii, direc in enumerate(['x', 'y', 'z']):
            if self.direction == direc:
                assert self.widths[ii] == 0.

        # Pre-select a narrow region around the region-of-interest
        thickness = 4.0 * np.cbrt((snap["0_Volume"]) / (4.0 * np.pi / 3.0))

        self.slice = util.get_index_of_slice_region(snap["0_Coordinates"], center, widths,
                                                    thickness, snap.box_size)

        self.index_in_slice_region = np.arange(snap["0_Coordinates"].shape[0])[self.slice]

        # Construct a tree
        self.pos = snap["0_Coordinates"][self.slice]
        tree = KDTree(self.pos)

        # Now construct the image grid
        w, h = self._get_width_and_height_arrays()

        center = self.center
        ones = np.ones(w.shape[0])
        if direction == 'x':
            image_points = np.vstack([ones * center[0], w, h]).T
        elif direction == 'y':
            image_points = np.vstack([w, ones * center[1], h]).T
        elif direction == 'z':
            image_points = np.vstack([w, h, ones * center[2]]).T

        # Query the tree to obtain closest Voronoi cell indices
        d, i = tree.query(image_points, workers=settings.numthreads)

        self.index = self._unflatten(self.index_in_slice_region[i])
        self.distance_to_nearest_cell = self._unflatten(d)

    def _get_width_and_height_arrays(self):
        """
        Get width and height coordinates in the image as 1D arrays
        of total length npix_width × npix_height.
        """
        extent = self.extent

        self.npix_width = npix_width = self.npix
        width = extent[1] - extent[0]
        height = extent[3] - extent[2]

        # TODO: Make assertion that dx=dy
        self.npix_height = npix_height = int(height / width * npix_width)

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

    def slice_variable(self, variable):
        """
        Slice a gas variable based on the Voronoi cells closest to the image
        plane.

        Parameters
        ----------
        variable: a string or an array of shape (N, )
                  representing the gas variable to slice

        Returns:
        An array of shape (npix, npix) representing the sliced gas variable
        """

        if isinstance(variable, str):
            variable = self.snap[variable]
        else:
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        return variable[self.index]
