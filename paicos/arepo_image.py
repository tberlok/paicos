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
    def __init__(self, image_filename, snap=None, center=None, widths=None,
                 direction=None, image_creator=None, mode='w'):

        """
        The filenames should include the full path:

        image_filename: e.g. "./projections/thin_projection_247_x.hdf5"

        snap: an instance of the Snapshot class

        If your image was created using a Paicos Projector or Slicer object. You can
        pass such an object using the image_creator input. Alternatively,
        you can simply

        center: A length-3 array giving the center of the image.

        widths: This is a length-3 array giving the widths of
                the image. For slices, the value indicating the thickness
                can be set to zero.

        direction: the viewing direction. Set this to e.g. 'x', 'y' or 'z'.

        """

        if image_creator is not None:
            self.center = np.array(image_creator.center)
            self.widths = np.array(image_creator.widths)
            self.direction = image_creator.direction
        elif (center is not None) and (widths is not None) and (direction is not None):
            self.center = center
            self.widths = widths
            self.direction = direction
        else:
            msg = """
            Either pass a projector/slicer object to image_creator,
            or alternatively, specify 'center', 'widths' and 'direction'
            of your image"""
            raise RuntimeError(msg)

        self.xc = self.center[0]
        self.yc = self.center[1]
        self.zc = self.center[2]

        self.width_x = self.widths[0]
        self.width_y = self.widths[1]
        self.width_z = self.widths[2]

        self.mode = mode

        if self.direction == 'x':
            self.extent = [self.yc - self.width_y/2, self.yc + self.width_y/2,
                           self.zc - self.width_z/2, self.zc + self.width_z/2]

        elif self.direction == 'y':
            self.extent = [self.xc - self.width_x/2, self.xc + self.width_x/2,
                           self.zc - self.width_z/2, self.zc + self.width_z/2]

        elif self.direction == 'z':
            self.extent = [self.xc - self.width_x/2, self.xc + self.width_x/2,
                           self.yc - self.width_y/2, self.yc + self.width_y/2]

        self.image_filename = image_filename
        tmp_list = image_filename.split('/')
        tmp_list[-1] = 'tmp_' + tmp_list[-1]
        self.tmp_image_filename = ''
        for ii in range(0, len(tmp_list)-1):
            self.tmp_image_filename += tmp_list[ii] + '/'
        self.tmp_image_filename += tmp_list[-1]
        self.arepo_snap_filename = snap.first_snapfile_name

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

    def save_image(self, name, data, attrs=None):
        """
        This function saves a 2D image to the hdf5 file.
        """
        with h5py.File(self.tmp_image_filename, 'r+') as f:
            f.create_dataset(name, data=data)

    def finalize(self):
        """
        """
        import os
        if self.mode == 'w':
            os.rename(self.tmp_image_filename, self.image_filename)
        elif self.mode == 'a' or self.mode == 'r+':
            with h5py.File(self.tmp_image_filename, 'r') as tmp:
                with h5py.File(self.image_filename, 'r+') as final:
                    np.testing.assert_array_equal(tmp['image_info'].attrs['center'],
                                                  final['image_info'].attrs['center'])
                    np.testing.assert_array_equal(tmp['image_info'].attrs['widths'],
                                                  final['image_info'].attrs['widths'])
                    assert tmp['image_info'].attrs['direction'] == final['image_info'].attrs['direction']
                    assert tmp['Header'].attrs['Time'] == final['Header'].attrs['Time']
                    for key in tmp.keys():
                        if key not in final.keys():
                            final.create_dataset(key, data=tmp[key])
            os.remove(self.tmp_image_filename)


if __name__ == '__main__':
    from paicos import root_dir
    from paicos import Snapshot
    from paicos import Projector

    image_filename = root_dir + "/data/test_arepo_image_format.hdf5"

    snap = Snapshot(root_dir + '/data/', 247)

    # A length-3 array giving the center of the image.
    center = [250000, 400000, 500000]

    # This is a length-3 array giving the widths of the image.
    widths = [10000, 10000, 2000]

    # The viewing direction. Set this to e.g. 'x', 'y' or 'z'.
    direction = 'z'

    # Create arepo image file.
    # The file will have 'tmp_' prepended to the filename until .finalize()
    # is called.
    image_file = ArepoImage(image_filename, snap, center, widths, direction)

    # Save some images to the file (in a real example one would first import\\
    # and use a projection function)
    image_file.save_image('random_data', np.random.random((200, 200)))
    image_file.save_image('random_data2', np.random.random((400, 400)))

    # Move from temporary filename to final filename
    image_file.finalize()

    # Now amend the file with another set of data
    p = Projector(snap, center, widths, direction)
    image_file = ArepoImage(image_filename, snap, image_creator=p, mode='a')
    image_file.save_image('random_data3', np.random.random((500, 500)))
    image_file.finalize()

    with h5py.File(image_filename, 'r') as f:
        print(list(f.keys()))
