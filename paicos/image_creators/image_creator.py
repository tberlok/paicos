"""
This defines a base class for image creators (such as Projector and Slicer)
"""

import numpy as np
from .. import util
from .. import settings
from .. import units


class ImageCreator:
    """
    This is a base class for creating images from a snapshot.
    """

    def __init__(self, snap, center, widths, direction, npix=512):
        """
        Initialize the ImageCreator class. This method will be called
        by subclasses such as Projector or Slicer.

        Parameters:
            snap (object): Snapshot object from which the image is created

            center: Center of the image (3D coordinates)

            widths: Width of the image in each direction (3D coordinates)

            direction (str): Direction of the image ('x', 'y', 'z')

            npix (int): Number of pixels in the image (default is 512)
        """

        self.snap = snap

        code_length = self.snap.length

        if hasattr(center, 'unit'):
            self.center = center
            assert center.unit == code_length.unit
        elif settings.use_units:
            self.center = np.array(center) * code_length
        else:
            self.center = np.array(center)

        if hasattr(widths, 'unit'):
            self.widths = widths
            assert widths.unit == code_length.unit
        elif settings.use_units:
            self.widths = np.array(widths) * code_length
        else:
            self.widths = np.array(widths)

        self.x_c = self.center[0]
        self.y_c = self.center[1]
        self.z_c = self.center[2]
        self.width_x = self.widths[0]
        self.width_y = self.widths[1]
        self.width_z = self.widths[2]

        self.direction = direction

        self.npix = npix

        if direction == 'x':
            self.extent = [self.y_c - self.width_y / 2, self.y_c + self.width_y / 2,
                           self.z_c - self.width_z / 2, self.z_c + self.width_z / 2]

        elif direction == 'y':
            self.extent = [self.x_c - self.width_x / 2, self.x_c + self.width_x / 2,
                           self.z_c - self.width_z / 2, self.z_c + self.width_z / 2]

        elif direction == 'z':
            self.extent = [self.x_c - self.width_x / 2, self.x_c + self.width_x / 2,
                           self.y_c - self.width_y / 2, self.y_c + self.width_y / 2]

        if settings.use_units:
            self.extent = units.PaicosQuantity(self.extent, a=snap._Time, h=snap.h,
                                               comoving_sim=snap.comoving_sim)
        else:
            self.extent = np.array(self.extent)

        area = (self.extent[1] - self.extent[0]) * (self.extent[3] - self.extent[2])
        self.area = area
