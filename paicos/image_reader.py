import h5py
from .arepo_converter import ArepoConverter
from . import util
from .paicos_io import PaicosReader


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
                # Keys of the form 'MagneticFieldSquaredTimesVolumes'
                # are split up
                start, end = key.split('Times')
                if (end in keys):
                    self[start] = self[key]/self[end]
                elif (start[0:2] + end in keys):
                    self[start] = self[key]/self[start[0:2] + end]

        # Calculate density if we have both masses and volumes
        for p in ['', '0_']:
            if (p + 'Masses' in keys) and (p + 'Volumes' in keys):
                self[p + 'Density'] = self[p+'Masses']/self[p+'Volumes']
