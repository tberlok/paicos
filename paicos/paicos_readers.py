from . import util
import h5py
import os


class PaicosReader(dict):

    def __init__(self, basedir='.', snapnum=None, basename="snap",
                 load_all=True, to_physical=False,
                 basesubdir='snapdir', verbose=False):

        if basedir[-1] != '/':
            basedir += '/'

        if snapnum is None:
            self.filename = basedir + basename + '.hdf5'
            msg = 'File: {} not found found'.format(self.filename)
            assert os.path.exists(self.filename), msg
        else:
            single_file = basename + "_{:03d}.hdf5"
            multi_file = basesubdir + '_{:03d}/' + basename + '_{:03d}.{}.hdf5'
            multi_wo_dir = basename + '_{:03d}.{}.hdf5'
            #

            single_file = basedir + single_file.format(snapnum)
            multi_file = basedir + multi_file.format(snapnum, snapnum, '{}')
            multi_wo_dir = basedir + multi_wo_dir.format(snapnum, '{}')

            if os.path.exists(single_file):
                self.multi_file = False
                self.filename = single_file
            elif os.path.exists(multi_file.format(0)):
                self.multi_file = True
                self.first_file_name = self.filename = multi_file.format(0)
                self.no_subdir = False
            elif os.path.exists(multi_wo_dir.format(0)):
                self.multi_file = True
                self.first_file_name = self.filename = multi_wo_dir.format(0)
                self.no_subdir = True
            else:
                err_msg = "File: {} not found found"
                raise FileNotFoundError(err_msg.format(self.file_name))

        with h5py.File(self.filename, 'r') as f:
            self.Header = dict(f['Header'].attrs)
            self.Config = dict(f['Config'].attrs)
            self.Parameters = dict(f['Parameters'].attrs)
            keys = list(f.keys())

        # self.converter = ArepoConverter(self.filename)

        # self.h = self.converter.h
        # self.scale_factor = self.a = self.converter.a

        # if self.converter.ComovingIntegrationOn:
            # self.redshift = self.z = self.converter.z
            # self.age = self.converter.age
            # self.lookback_time = self.converter.lookback_time
        # else:
            # self.time = self.converter.time

        # Load all data sets
        # if load_all:
        #     for key in keys:
        #         self.load_data(key)

    def load_data(self, name, group=None):

        with h5py.File(self.filename, 'r') as f:
            if isinstance(f[name], h5py._hl.dataset.Dataset):
                self[name] = util.load_dataset(f, name, group=group,
                                               converter=self.converter)
            elif isinstance(f[name], h5py._hl.group.Group):
                for data_name in f[name].keys():
                    data = util.load_dataset(f, data_name, group=name,
                                             converter=self.converter)
                    if name not in self:
                        self[name] = {}
                    self[name][data_name] = data


class Catalog2(PaicosReader):
    def __init__(self, basedir='.', snapnum=None, load_all=True,
                 to_physical=False, subfind_catalog=True, verbose=False):

        if subfind_catalog:
            basename = 'fof_subhalo_tab'
        else:
            basename = 'fof_tab'

        super().__init__(basedir=basedir, snapnum=snapnum, basename=basename,
                         load_all=load_all, to_physical=to_physical,
                         basesubdir='groups', verbose=verbose)




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
