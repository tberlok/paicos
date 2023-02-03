import h5py
import numpy as np
from .import util
from .paicos_writer import PaicosWriter
from . import settings
from . import units


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

            numthreads (int): Number of threads to use (default is 1)
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
            self.extent = units.PaicosQuantity(self.extent, a=snap.a, h=snap.h,
                                               comoving_sim=snap.comoving_sim)
        else:
            self.extent = np.array(self.extent)

        area = (self.extent[1] - self.extent[0]) * (self.extent[3] - self.extent[2])
        self.area = area


class ArepoImage(PaicosWriter):
    """
    A derived data format for Arepo snapshots.

    A common task is to reduce Arepo snapshots to 2D arrays (either slices or
    projections) which can plotted as images. The computation of the 2D arrays
    can be time-consuming for high-resolution simulations with large
    snapshots. The purpose of this class is to define a derived data format
    which can be used to store images for later plotting with matplotlib.

    """

    def __init__(self, image_creator, basedir, basename="projection",
                 mode='w'):
        """
        If your image was created using a Paicos Projector or Slicer object,
        then you can pass such an object using the image_creator input
        argument.

        basedir (file path): folder where you would like to save the image
                             file.

        basename (string): the file will have a name like "projection_{:03d}"
                           where {:03d} is automatically replaced with the
                           snapnum.

        """
        self.center = image_creator.center
        self.widths = image_creator.widths
        self.extent = image_creator.extent
        self.direction = image_creator.direction

        # This creates an image at self.tmp_filename (if mode='w')
        super().__init__(image_creator.snap, basedir, basename=basename,
                         mode=mode)

        # Create image file and write information about image
        if self.mode == 'w':
            with h5py.File(self.tmp_filename, 'r+') as file:
                util.save_dataset(file, 'center', self.center, group='image_info')
                util.save_dataset(file, 'widths', self.widths, group='image_info')
                util.save_dataset(file, 'extent', self.extent, group='image_info')
                file['image_info'].attrs['direction'] = self.direction
                file['image_info'].attrs['image_creator'] = str(image_creator)

    def save_image(self, name, data):
        """
        This function saves a 2D image to the hdf5 file.
        """
        self.write_data(name, data)

    def perform_extra_consistency_checks(self):
        with h5py.File(self.filename, 'r') as f:
            center = util.load_dataset(f, 'center', group='image_info')
            if settings.use_units:
                assert center.unit == self.center.unit
                np.testing.assert_array_equal(center.value, self.center.value)
            else:
                np.testing.assert_array_equal(center, self.center)

            widths = util.load_dataset(f, 'widths', group='image_info')
            if settings.use_units:
                assert widths.unit == self.widths.unit
                np.testing.assert_array_equal(widths.value, self.widths.value)
            else:
                np.testing.assert_array_equal(widths, self.widths)

            assert f['image_info'].attrs['direction'] == self.direction
