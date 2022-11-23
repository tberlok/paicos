import h5py
import numpy as np
from paicos import ComovingQuantity, ArepoConverter


class ImageReader(dict):

    def __init__(self, basedir, snapnum, basename="projection", verbose=False):
        pass

        #
        self.filename = basedir + basename + '_{:03d}.hdf5'.format(snapnum)

        with h5py.File(self.filename, 'r') as f:
            self.Header = dict(f['Header'].attrs)
            self.Config = dict(f['Config'].attrs)
            self.Param = dict(f['Parameters'].attrs)
            keys = list(f.keys())

        self.Redshift = self.Header['Redshift']
        self.h = self.Header['HubbleParam']
        self.scale_factor = self.Header['Time']
        # self.time (as an astropy object in Gyr)
        self.converter = ArepoConverter(self.filename)
        with h5py.File(self.filename, 'r') as f:
            get_func = self.converter.get_comoving_quantity
            self.extent = get_func('Coordinates',
                                   f['image_info'].attrs['extent'])
            self.widths = get_func('Coordinates',
                                   f['image_info'].attrs['widths'])
            self.center = get_func('Coordinates',
                                   f['image_info'].attrs['center'])

        # Load all data sets
        for key in keys:
            self.load_data(key)

    def load_data(self, name):
        get_func = self.converter.get_comoving_quantity
        with h5py.File(self.filename, 'r') as f:
            if isinstance(f[name], h5py._hl.dataset.Dataset):
                data = f[name][...]
                attrs = dict(f[name].attrs)
                if len(attrs) > 0:
                    self[name] = get_func(attrs, data)
                else:
                    self[name] = get_func(name, data)


if __name__ == '__main__':
    from paicos import root_dir

    im = ImageReader(root_dir + '/data/', 247,
                     basename='test_arepo_image_format')
