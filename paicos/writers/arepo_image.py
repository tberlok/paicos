"""
This defines a class for saving images as hdf5 files in a systematic way.
"""

import h5py
import numpy as np
from .. import util
from .paicos_writer import PaicosWriter
from .. import settings
from .. import units
from ..orientation import Orientation


class ArepoImage(PaicosWriter):
    """
    A derived data format for Arepo snapshots.

    A common task is to reduce Arepo snapshots to 2D arrays (either slices or
    projections) which can plotted as images. The computation of the 2D arrays
    can be time-consuming for high-resolution simulations with large
    snapshots. The purpose of this class is to define a derived data format
    which can be used to store images for later plotting with matplotlib.
    """

    def __init__(self, image_creator, basedir, basename="projection", mode='w'):
        """
        Initialize an HDF5 file for storing an image.

        This class is intended to be used with images created using a Paicos
        Projector or Slicer object, which can be passed as the `image_creator`
        input argument.

        The image data, including 'center', 'widths', 'extent', and
        'direction', will be extracted from the `image_creator` object and
        stored in the HDF5 file. The HDF5 file is created in the `basedir`
        folder, with the name `basename_{:03d}`.

        The `mode` argument controls whether the file is opened in write mode
        ('w') or amend mode ('a'). If `mode` is 'w', the file will be created
        at `self.tmp_filename` and the image information will be written to
        the file. Setting the mode to amend mode, 'a', allows to add new images
        to an already existing file.

        Parameters
        ----------

            image_creator : object
                A Paicos Projector or Slicer object used to create the image.

            basedir (file: path
                The folder where the image file should be saved.

            basename : string
                The base name for the image file, which will be in
                the format ``basename_{:03d}``. (default: "projection").

            mode : string
                The mode to open the file in, either 'w' for write mode
                or 'a' for append mode. (default: 'w').

        Methods
        -------

            finalize :
                Changes the filename from a temporary one to the final one,
                e.g., from self.tmp_filename to self.filename.
        """

        self.center = image_creator.center
        self.widths = image_creator.widths
        self.extent = image_creator.extent
        self.direction = image_creator.direction
        self.orientation = image_creator.orientation

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
                file['image_info'].attrs['npix_width'] = image_creator.npix_width
                file['image_info'].attrs['npix_height'] = image_creator.npix_height

                if self.direction == 'orientation':
                    file['image_info'].attrs['normal_vector'] = self.orientation.normal_vector
                    file['image_info'].attrs['perp_vector1'] = self.orientation.perp_vector1

    def save_image(self, name, data):
        """
        This method saves a 2D image to the hdf5 file.
        """
        self.write_data(name, data)

    def _perform_extra_consistency_checks(self):
        """
        Perform extra consistency checks on the HDF5 file to ensure the
        current values match the values stored in the HDF5 file that we intend
        to amend.

        The function opens the HDF5 file, loads the 'center' and 'widths'
        datasets from the 'image_info' group, and compares them to the
        corresponding values stored in memory. The 'direction' attribute of
        the 'image_info' group is also checked for consistency.
        """
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


ImageWriter = ArepoImage
