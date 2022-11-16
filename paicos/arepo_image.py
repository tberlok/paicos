import h5py
import numpy as np


class ArepoImage:
    """
    A derived data format for Arepo snapshots.

    A common task is to reduce Arepo snapshots to 2D arrays (either slices or
    projections) which can plotted as images. The computation of the 2D arrays
    can be time-consuming for high-resolution simulations with large
    snapshots. The purpose of this class is to define a derived data format
    which can be used to store images for later plotting with matplotlib.

    The creation of the 2D array is decoupled from this class, so that any
    custom method may be used (e.g. arepo-snap-util).

    """
    def __init__(self, image_filename, arepo_snap_filename,
                 center, widths, direction):

        """
        The filenames should include the full path:

        image_filename: e.g. "./projections/thin_projection_247_x.hdf5"

        arepo_snap_filename: e.g. "./output/snapdir_247/snap_247.0.hdf5"

        center: A length-3 array giving the center of the image.

        widths: This is a length-3 array giving the widths of
                the image. For slices, the value indicating the thickness
                can be set to zero.

        direction: the viewing direction. Set this to e.g. 'x', 'y' or 'z'.
        """

        self.center = center
        self.xc = center[0]
        self.yc = center[1]
        self.zc = center[2]

        self.widths = widths
        self.width_x = widths[0]
        self.width_y = widths[1]
        self.width_z = widths[2]

        self.direction = direction

        if direction == 'x':
            self.extent = [self.yc - self.width_y/2, self.yc + self.width_y/2,
                           self.zc - self.width_z/2, self.zc + self.width_z/2]

        elif direction == 'y':
            self.extent = [self.xc - self.width_x/2, self.xc + self.width_x/2,
                           self.zc - self.width_z/2, self.zc + self.width_z/2]

        elif direction == 'z':
            self.extent = [self.xc - self.width_x/2, self.xc + self.width_x/2,
                           self.yc - self.width_y/2, self.yc + self.width_y/2]

        self.image_filename = image_filename
        tmp_list = image_filename.split('/')
        tmp_list[-1] = 'tmp_' + tmp_list[-1]
        self.tmp_image_filename = ''
        for ii in range(0, len(tmp_list)-1):
            self.tmp_image_filename += tmp_list[ii] + '/'
        self.tmp_image_filename += tmp_list[-1]
        self.arepo_snap_filename = arepo_snap_filename

        self.create_projection_file()

        self.copy_over_snapshot_information()

        # Write information about image
        with h5py.File(self.tmp_image_filename, 'r+') as f:
            f.create_group('image_info')
            f['image_info'].attrs['center'] = self.center
            f['image_info'].attrs['widths'] = self.widths
            f['image_info'].attrs['direction'] = self.direction
            f['image_info'].attrs['extent'] = self.extent

    def create_projection_file(self):
        """
        Here we create the image file and store information about the
        image dimensions.
        """

        with h5py.File(self.tmp_image_filename, 'w') as f:
            f.attrs['center'] = self.center
            f.attrs['xc'] = self.xc
            f.attrs['yc'] = self.yc
            f.attrs['zc'] = self.zc

            f.attrs['widths'] = self.widths
            f.attrs['width_x'] = self.width_x
            f.attrs['width_y'] = self.width_y
            f.attrs['width_z'] = self.width_z

            f.attrs['direction'] = self.direction

            if self.direction == 'x':
                f.attrs['thickness'] = self.width_x
                f.attrs['extent'] = [self.yc - self.width_y/2,
                                     self.yc + self.width_y/2,
                                     self.zc - self.width_z/2,
                                     self.zc + self.width_z/2]
                if self.width_y == self.width_z:
                    f.attrs['sidelength'] = self.width_y

            elif self.direction == 'y':
                f.attrs['thickness'] = self.width_y
                f.attrs['extent'] = [self.xc - self.width_x/2,
                                     self.xc + self.width_x/2,
                                     self.zc - self.width_z/2,
                                     self.zc + self.width_z/2]
                if self.width_x == self.width_z:
                    f.attrs['sidelength'] = self.width_x

            elif self.direction == 'z':
                f.attrs['thickness'] = self.width_z
                f.attrs['extent'] = [self.xc - self.width_x/2,
                                     self.xc + self.width_x/2,
                                     self.yc - self.width_y/2,
                                     self.yc + self.width_y/2]
                if self.width_x == self.width_y:
                    f.attrs['sidelength'] = self.width_y

    def copy_over_snapshot_information(self):
        """
        Copy over attributes from the original arepo snapshot.
        In this way we will have access to units used, redshift etc
        """
        g = h5py.File(self.arepo_snap_filename, 'r')
        with h5py.File(self.tmp_image_filename, 'r+') as f:
            for group in ['Header', 'Parameters', 'Config']:
                f.create_group(group)
                for key in g[group].attrs.keys():
                    f[group].attrs[key] = g[group].attrs[key]
        g.close()

    def save_image(self, name, data):
        """
        This function saves a 2D image to the hdf5 file.
        """
        with h5py.File(self.tmp_image_filename, 'r+') as f:
            f.create_dataset(name, data=data)

    def finalize(self):
        """
        # TODO: Overload an out-of-scope operator instead?
        """
        import os
        os.rename(self.tmp_image_filename, self.image_filename)


if __name__ == '__main__':
    from paicos import get_project_root_dir

    path = get_project_root_dir()
    image_filename = path + "/data/test_arepo_image_format.hdf5"

    arepo_snap_filename = path + "/data/snap_247.hdf5"

    # A length-3 array giving the center of the image.
    center = [250000, 400000, 500000]

    # This is a length-3 array giving the widths of the image.
    widths = [10000, 10000, 2000]

    # The viewing direction. Set this to e.g. 'x', 'y' or 'z'.
    direction = 'z'

    # Create arepo image file.
    # The file will have 'tmp_' prepended to the filename until .finalize()
    # is called.
    image_file = ArepoImage(image_filename, arepo_snap_filename,
                            center, widths, direction)

    # Save some images to the file (in a real example one would first import\\
    # and use a projection function)
    image_file.save_image('random_data', np.random.random((200, 200)))
    image_file.save_image('random_data2', np.random.random((400, 400)))

    # Move from temporary filename to final filename
    image_file.finalize()
