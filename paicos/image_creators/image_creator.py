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

    def __init__(self, snap, center, widths, direction, npix=512, parttype=0):
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

        self.npix = self.npix_width = npix

        self.parttype = parttype

        if direction == 'x':
            extent = [self.y_c - self.width_y / 2, self.y_c + self.width_y / 2,
                      self.z_c - self.width_z / 2, self.z_c + self.width_z / 2]

            centered_extent = [- self.width_y / 2, self.width_y / 2,
                               - self.width_z / 2, self.width_z / 2]

            self.width = self.width_y
            self.height = self.width_z
            self.depth = self.width_x

        elif direction == 'y':
            extent = [self.x_c - self.width_x / 2, self.x_c + self.width_x / 2,
                      self.z_c - self.width_z / 2, self.z_c + self.width_z / 2]

            centered_extent = [- self.width_x / 2, self.width_x / 2,
                               - self.width_z / 2, self.width_z / 2]

            self.width = self.width_x
            self.height = self.width_z
            self.depth = self.width_y

        elif direction == 'z':
            extent = [self.x_c - self.width_x / 2, self.x_c + self.width_x / 2,
                      self.y_c - self.width_y / 2, self.y_c + self.width_y / 2]

            centered_extent = [- self.width_x / 2, self.width_x / 2,
                               - self.width_y / 2, self.width_y / 2]

            self.width = self.width_x
            self.height = self.width_y
            self.depth = self.width_z

        if settings.use_units:
            a = snap._Time
            h = snap.h
            comoving = snap.comoving_sim
            self.extent = units.PaicosQuantity(extent, a=a, h=h, comoving_sim=comoving)

            self.centered_extent = units.PaicosQuantity(centered_extent, a=a, h=h,
                                                        comoving_sim=comoving)
            self.npix_height = int((self.height / self.width).value * self.npix_width)
        else:
            self.extent = np.array(extent)
            self.centered_extent = np.array(centered_extent)
            self.npix_height = int((self.height / self.width) * self.npix_width)

        area = (self.extent[1] - self.extent[0]) * (self.extent[3] - self.extent[2])
        self.area = area
        self.area_per_pixel = self.area / (self.npix_width * self.npix_height)
        self.volume = self.width_x * self.width_y * self.width_z
        self.volume_per_pixel = self.volume / (self.npix_width * self.npix_height)
