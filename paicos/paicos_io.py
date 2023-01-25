from .arepo_converter import ArepoConverter
from . import util
import h5py


class PaicosReader(dict):

    def __init__(self, basedir, snapnum, basename="projection", load_all=True):

        if basedir[-1] != '/':
            basedir += '/'
        #
        self.filename = basedir + basename + '_{:03d}.hdf5'.format(snapnum)

        with h5py.File(self.filename, 'r') as f:
            self.Header = dict(f['Header'].attrs)
            self.Config = dict(f['Config'].attrs)
            self.Parameters = dict(f['Parameters'].attrs)
            keys = list(f.keys())

        self.converter = ArepoConverter(self.filename)

        self.redshift = self.z = self.converter.z
        self.h = self.converter.h
        self.scale_factor = self.a = self.converter.a

        if self.Parameters['ComovingIntegrationOn'] == 1:
            self.age = self.converter.age
            self.lookback_time = self.converter.lookback_time
        else:
            self.time = self.converter.time

        # Load all data sets
        if load_all:
            for key in keys:
                self.load_data(key)

    def load_data(self, name, group=None):

        with h5py.File(self.filename, 'r') as f:
            if isinstance(f[name], h5py._hl.dataset.Dataset):
                self[name] = util.load_dataset(f, name, group=group,
                                               converter=self.converter)


class ImageReader(PaicosReader):

    def __init__(self, basedir, snapnum, basename="projection", load_all=True):

        # The PaicosReader class takes care of most of the loading
        super().__init__(basedir, snapnum, basename=basename,
                         load_all=load_all)

        # Load info specific to images
        with h5py.File(self.filename, 'r') as f:
            self.extent = util.load_dataset(f, 'extent', group='image_info')
            self.widths = util.load_dataset(f, 'widths', group='image_info')
            self.center = util.load_dataset(f, 'center', group='image_info')
            self.direction = f['image_info'].attrs['direction']
            self.image_creator = f['image_info'].attrs['image_creator']

        # Get derived images from projection-files
        keys = list(self.keys())
        for key in keys:
            if 'Times' in key:
                # Keys of the form 'MagneticFieldSquaredTimesVolume'
                # are split up
                start, end = key.split('Times')
                if (end in keys):
                    self[start] = self[key]/self[end]
                elif (start[0:2] + end in keys):
                    self[start] = self[key]/self[start[0:2] + end]

        # Calculate density if we have both masses and volumes
        for p in ['', '0_']:
            if (p + 'Masses' in keys) and (p + 'Volume' in keys):
                self[p + 'Density'] = self[p+'Masses']/self[p+'Volume']


class Histogram2DReader(PaicosReader):

    def __init__(self, basedir, snapnum, basename='2d_histogram'):

        # The PaicosReader class takes care of most of the loading
        super().__init__(basedir, snapnum, basename=basename,
                         load_all=True)

        with h5py.File(self.filename, 'r') as hdf5file:
            if 'colorlabel' in hdf5file['hist2d'].attrs.keys():
                self.colorlabel = hdf5file['hist2d'].attrs['colorlabel']
            self.normalize = hdf5file['hist_info'].attrs['normalize']
            self.logscale = hdf5file['hist_info'].attrs['normalize']

        self.hist2d = self['hist2d']
        self.centers_x = self['centers_x']
        self.centers_y = self['centers_y']
