"""
This defines a base class for image creators (such as Projector and Slicer)
"""

import numpy as np
import warnings
from .. import util
from .. import settings
from .. import units
from ..orientation import Orientation


class ImageCreator:
    """
    This is a base class for creating images from a snapshot.
    """

    def __init__(self, snap, center, widths, direction, npix=512, parttype=0):
        """
        Initialize the ImageCreator class. This method will be called
        by subclasses such as Projector or Slicer.

        Parameters:
            snap (object): Snapshot object from which the image is created.

            center: Center of the image (3D coordinates).

            widths: Width of the image in each direction (3D coordinates)

            direction (str, Orientation): Direction of the image ('x', 'y', 'z')
                or an Orientation instance.

            npix (int): Number of pixels in the image (default is 512).
        """

        self._obervers = []

        self.snap = snap

        if settings.use_units:
            code_length = self.snap.length

        if hasattr(center, 'unit'):
            self._center = center.copy
            assert center.unit == code_length.unit, 'this restriction applies'
        elif settings.use_units:
            self._center = np.array(center) * code_length
        else:
            self._center = np.array(center)

        # Difference from original center of image
        if settings.use_units:
            self._diff_center = self._center.copy * 0
        else:
            self._diff_center = np.array(self._center) * 0

        if hasattr(widths, 'unit'):
            self._widths = widths.copy
            assert widths.unit == code_length.unit, 'this restriction applies'
        elif settings.use_units:
            self._widths = np.array(widths) * code_length
        else:
            self._widths = np.array(widths)

        if isinstance(direction, str):

            self.direction = direction
            if direction == 'x':
                self.orientation = Orientation(normal_vector=[1, 0, 0],
                                               perp_vector1=[0, 1, 0])
            elif direction == 'y':
                warnings.warn("\n\nThe definition of a 'y'-image was changed "
                              + "on 19/1/24 to be consistent with a right-handed "
                              + "coordinate system. You can recover the old "
                              + "behaviour by passing an orientation object:\n"
                              + "Orientation(normal_vector=[0, -1, 0], perp_vector1=[1, 0, 0])\n"
                              + "instead of a direction 'y' string.\n\n")
                self.orientation = Orientation(normal_vector=[0, 1, 0],
                                               perp_vector1=[0, 0, 1])
            elif direction == 'z':
                self.orientation = Orientation(normal_vector=[0, 0, 1],
                                               perp_vector1=[1, 0, 0])

        elif isinstance(direction, Orientation):
            self.direction = 'orientation'
            self.orientation = direction

        self._npix = self._npix_width = npix

        self._parttype = parttype

        # For checking if properties changes
        self._old_orientation = self.orientation.copy
        self._properties_changed = False

        # Check that units are consistent
        self.do_unit_consistency_check()

        # Check if image region exceeds simulation domain
        # TODO

    def _check_if_properties_changed(self):

        # Check if orientation changed compared to last
        # time _do_region_selection was called
        orientation_changed = not self.orientation._are_equal(self._old_orientation)

        if orientation_changed:
            self.direction == 'orientation'

        # print('_check_if_properties_changed called')
        if self._properties_changed or orientation_changed:
            self._do_region_selection()
            self._properties_changed = False
            self._old_orientation = self.orientation.copy
            return True
        else:
            return False

    def do_unit_consistency_check(self):
        """
        This function is called inside image creators to ensure
        that length units are consistent.
        """

        if settings.use_units:
            if hasattr(self, 'hsml'):
                assert self.hsml.unit.is_equivalent(self.widths.unit)
            if hasattr(self, 'pos'):
                assert self.pos.unit == self.widths.unit, (self.pos.unit, self.widths.unit)
            assert self.center.unit == self.widths.unit, (self.center.unit, self.widths.unit)
            if f"{self.parttype}_Coordinates" in self.snap:
                assert self.center.unit == self.snap[f"{self.parttype}_Coordinates"].unit
            if hasattr(self.snap, 'box_size'):
                assert self.center.unit == self.snap.box_size.unit

    def _do_region_selection(self):
        err_msg = ("_do_region_selection was called from ImageCreator. "
                   + "This should never happen as the Subclasses of "
                   + "ImageCreator should implement this method")
        raise RuntimeError(err_msg)

    def info(self):
        """
        Prints some information about the image creator instance.
        TODO: Add things like the number of cells/particles
        in the current instance.
        """
        s = f'Paicos ImageCreator instance of type: {str(self)}\n'
        s += f'widths: {self.widths}\n'
        s += f'center:  {self.center}\n'
        s += f'orientation:  {self.orientation}'
        print(s)

    @property
    def center(self):
        """
        Center of the image (3D coordinates).
        """
        return self._center

    @property
    def widths(self):
        """
        Width of the image in each direction (3D coordinates).
        """
        return self._widths

    @widths.setter
    def widths(self, value):
        """
        TODO: ensure that nx, ny
        remain consistent with widths
        """
        assert value.shape[0] == 3
        if settings.use_units:
            assert hasattr(value, 'unit')
            self._widths = value.copy
        else:
            self._widths = np.array(value)
        self._properties_changed = True

    @center.setter
    def center(self, value):
        assert value.shape[0] == 3
        if settings.use_units:
            assert hasattr(value, 'unit')
            self._center = value.copy
        else:
            self._center = np.array(value)
        self._properties_changed = True

    def reset_center(self):
        """
        Resets the center of the image creator to the
        value it had at initialization.
        """
        self.center = self.center - self._diff_center
        self._diff_center = np.zeros_like(self._diff_center)

    @property
    def npix(self):
        """
        The number of pixels along the horizontal
        direction of the image.
        """
        return self._npix

    @npix.setter
    def npix(self, value):
        self._npix = value
        # TODO: Not always needed to do this...
        self._properties_changed = True

    @property
    def npix_width(self):
        """
        The number of pixels along the horizontal
        direction of the image.
        """
        return self._npix

    @npix_width.setter
    def npix_width(self, value):
        self.npix = value

    @property
    def parttype(self):
        """
        The parttype being imaged, e.g. parttype=0
        are the Voronoi gas cells.
        """
        return self._parttype

    @property
    def x_c(self):
        """
        The x-component of .center
        """
        return self.center[0]

    @property
    def y_c(self):
        """
        The y-component of .center
        """
        return self.center[1]

    @property
    def z_c(self):
        """
        The z-component of .center
        """
        return self.center[2]

    @property
    def width_x(self):
        """
        The x-component of .widths
        """
        return self.widths[0]

    @property
    def width_y(self):
        """
        The y-component of .widths
        """
        return self.widths[1]

    @property
    def width_z(self):
        """
        The z-component of .widths
        """
        return self.widths[2]

    @property
    def extent(self):
        """
        The extent of the image in the horizontal-vertical plane.
        The idea with this property is that it corresponds to the
        matplotlib imshow keyword argument 'extent'. When using units,
        one has to remember to pass image_creator.extent.value rather
        than image_creator.extent.
        """
        if self.direction == 'orientation':

            # This is the extent of the image-plane pre-rotation
            # into the orientation of the image...
            # Use centered_extent for showing rotated images.
            center_width = self.x_c
            center_height = self.y_c

        elif self.direction == 'x':
            center_width = self.y_c
            center_height = self.z_c

        elif self.direction == 'y':
            center_width = self.z_c
            center_height = self.x_c

        elif self.direction == 'z':
            center_width = self.x_c
            center_height = self.y_c

        extent = [center_width - self.width / 2, center_width + self.width / 2,
                  center_height - self.height / 2, center_height + self.height / 2]

        if settings.use_units:
            extent = units.paicos_quantity_list_to_array(extent)
        else:
            extent = np.array(extent)
        return extent

    @property
    def centered_extent(self):
        """
        Same as 'extent' but centered on the center of the image.
        """
        centered_extent = [- self.width / 2, self.width / 2,
                           - self.height / 2, self.height / 2]

        if settings.use_units:
            centered_extent = units.paicos_quantity_list_to_array(centered_extent)
        else:
            centered_extent = np.array(centered_extent)
        return centered_extent

    @property
    def width(self):
        """
        The horizontal width of the image.
        """
        if self.direction == 'x':
            return self.width_y

        elif self.direction == 'y':
            return self.width_z

        elif self.direction == 'z' or self.direction == 'orientation':
            return self.width_x

    @property
    def height(self):
        """
        The vertical height of the image.
        """
        if self.direction == 'x':
            return self.width_z

        elif self.direction == 'y':
            return self.width_x

        elif self.direction == 'z' or self.direction == 'orientation':
            return self.width_y

    @property
    def depth(self):
        """
        The depth of the image.
        """
        if self.direction == 'x':
            return self.width_x

        elif self.direction == 'y':
            return self.width_y

        elif self.direction == 'z' or self.direction == 'orientation':
            return self.width_z

    @width.setter
    def width(self, value):
        if settings.use_units:
            assert hasattr(value, 'unit')
        if self.direction == 'x':
            self._widths[1] = np.copy(value)

        elif self.direction == 'y':
            self._widths[2] = np.copy(value)

        elif self.direction == 'z' or self.direction == 'orientation':
            self._widths[0] = np.copy(value)
        self._properties_changed = True

    @height.setter
    def height(self, value):
        if settings.use_units:
            assert hasattr(value, 'unit')
        if self.direction == 'x':
            self._widths[2] = np.copy(value)

        elif self.direction == 'y':
            self._widths[0] = np.copy(value)

        elif self.direction == 'z' or self.direction == 'orientation':
            self._widths[1] = np.copy(value)
        self._properties_changed = True

    @depth.setter
    def depth(self, value):
        if settings.use_units:
            assert hasattr(value, 'unit')
        if self.direction == 'x':
            self._widths[0] = np.copy(value)

        elif self.direction == 'y':
            self._widths[1] = np.copy(value)

        elif self.direction == 'z' or self.direction == 'orientation':
            self._widths[2] = np.copy(value)
        self._properties_changed = True

    def double_resolution(self):
        """
        Calling this method doubles the pixel resolution of the image
        creator.
        """
        self.npix = self.npix * 2

    def half_resolution(self):
        """
        Calling this method halves the pixel resolution of the image
        creator.
        """
        self.npix = self.npix // 2

    def zoom(self, factor):
        """
        Calling this method zooms the view by the input factor,
        that is, the horizontal width and vertical height of the
        image creator is changed like this:

        width = width / factor

        height = height / factor

        A factor less than one zooms out the view.
        """
        self.width = self.width / factor
        self.height = self.height / factor

        self._properties_changed = True

    def _move_center_along_unit_vector(self, shift, unit_vector):
        """
        Helper function for moving center along unit vector.
        """
        if settings.use_units:
            assert hasattr(shift, 'unit')

        # Calculate new center
        new_center = self._center + shift * unit_vector
        # Keep track of changes to center (offset from initialized center)
        self._diff_center += new_center - self._center

        # Store new center and set bool for re-calculation of image
        self._center = new_center
        self._properties_changed = True

    def move_center_along_normal_vector(self, shift):
        """
        Moves the center of the image along its depth direction.
        Value can be positive or negative and must have same units
        as the center.

        (TODO: In principle we could just convert the distance to same units).
        """
        self._move_center_along_unit_vector(shift, self.orientation.normal_vector)

    def move_center_along_perp_vector1(self, shift):
        """
        Moves the center of the image along its horizontal direction.
        Value can be positive or negative and must have same units
        as the center.

        (TODO: In principle we could just convert the distance to same units).
        """
        self._move_center_along_unit_vector(shift, self.orientation.perp_vector1)

    def move_center_along_perp_vector2(self, shift):
        """
        Moves the center of the image along its vertical direction.
        Value can be positive or negative and must have same units
        as the center.

        (TODO: In principle we could just convert the distance to same units).
        """
        self._move_center_along_unit_vector(shift, self.orientation.perp_vector2)

    @property
    def npix_height(self):
        """
        The number of pixels of an image in the vertical direction.
        """
        if settings.use_units:
            return int(round((self.height / self.width).value * self.npix_width))
        else:
            return int(round((self.height / self.width) * self.npix_width))

    @property
    def area(self):
        """
        The area of the image plane.
        """
        return (self.extent[1] - self.extent[0]) * (self.extent[3] - self.extent[2])

    @property
    def area_per_pixel(self):
        """
        The area per pixel of the image plane.
        """
        return self.area / (self.npix_width * self.npix_height)

    @property
    def volume(self):
        """
        The volume of the projection region (0 for slices).
        """
        return self.width * self.height * self.depth

    @property
    def volume_per_pixel(self):
        """
        The volume per pixel in the image.
        """
        return self.volume / (self.npix_width * self.npix_height)
