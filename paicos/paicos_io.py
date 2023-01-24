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
