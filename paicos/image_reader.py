import h5py
from .arepo_converter import ArepoConverter
from . import util


class ImageReader(dict):

    def __init__(self, basedir, snapnum, basename="projection", load_all=True,
                 verbose=False):

        if basedir[-1] != '/':
            basedir += '/'
        #
        self.filename = basedir + basename + '_{:03d}.hdf5'.format(snapnum)

        with h5py.File(self.filename, 'r') as f:
            self.Header = dict(f['Header'].attrs)
            self.Config = dict(f['Config'].attrs)
            self.Parameters = dict(f['Parameters'].attrs)
            keys = list(f.keys())

        self.redshift = self.z = self.Header['Redshift']
        self.h = self.Header['HubbleParam']
        self.scale_factor = self.a = self.Header['Time']
        self.converter = ArepoConverter(self.filename)

        if self.Parameters['ComovingIntegrationOn'] == 1:
            self.age = self.converter.age
            self.lookback_time = self.converter.lookback_time
        else:
            self.time = self.converter.time

        with h5py.File(self.filename, 'r') as f:
            self.extent = util.load_dataset(f, 'extent', group='image_info')
            self.widths = util.load_dataset(f, 'widths', group='image_info')
            self.center = util.load_dataset(f, 'center', group='image_info')
            self.direction = f['image_info'].attrs['direction']
            self.image_creator = f['image_info'].attrs['image_creator']

        # Load all data sets
        if load_all:
            for key in keys:
                self.load_data(key)

        # Get derived images from projection-files
        for key in keys:
            if 'Times' in key:
                # Keys of the form 'MagneticFieldSquaredTimesVolumes'
                # are split up
                start, end = key.split('Times')
                if (end in keys):
                    self[start] = self[key]/self[end]
                elif (start[0:2] + end in keys):
                    self[start] = self[key]/self[start[0:2] + end]

        for p in ['', '0_']:
            if (p + 'Masses' in keys) and (p + 'Volumes' in keys):
                self[p + 'Density'] = self[p+'Masses']/self[p+'Volumes']

    def load_data(self, name):
        get_func = self.converter.get_paicos_quantity
        with h5py.File(self.filename, 'r') as f:
            if isinstance(f[name], h5py._hl.dataset.Dataset):
                data = f[name][...]
                attrs = dict(f[name].attrs)
                if len(attrs) > 0:
                    self[name] = get_func(data, attrs)
                else:
                    self[name] = get_func(data, name)
