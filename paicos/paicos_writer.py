from .arepo_snap import Snapshot
from . import util
import h5py


class PaicosWriter:
    """
    """
    def __init__(self, snap_or_oject_containing_snap, basedir,
                 basename="paicos_file", mode='w'):

        if isinstance(snap_or_oject_containing_snap, Snapshot):
            self.snap = snap_or_oject_containing_snap
        else:
            self.snap = snap_or_oject_containing_snap.snap

        self.mode = mode

        snapnum = self.snap.snapnum

        if basedir[-1] != '/':
            basedir += '/'

        self.basedir = basedir
        self.basename = basename

        name = basename + '_{:03d}.hdf5'.format(snapnum)
        self.filename = basedir + name
        self.tmp_filename = basedir + 'tmp_' + name
        self.arepo_snap_filename = self.snap.first_snapfile_name

        if mode == 'w':
            self.copy_over_snapshot_information()
        else:
            self.perform_consistency_checks()

    def copy_over_snapshot_information(self):
        """
        Copy over attributes from the original arepo snapshot.
        In this way we will have access to units used, redshift etc
        """
        g = h5py.File(self.arepo_snap_filename, 'r')
        with h5py.File(self.tmp_filename, 'w') as f:
            for group in ['Header', 'Parameters', 'Config']:
                f.create_group(group)
                for key in g[group].attrs.keys():
                    f[group].attrs[key] = g[group].attrs[key]
        g.close()

    def write_data(self, name, data, group=None, group_attrs=None):

        if self.mode == 'w':
            filename = self.tmp_filename
        else:
            filename = self.filename

        f = h5py.File(filename, 'r+')

        if self.mode == 'a':
            msg = ('PaicosWriter is in amend mode but {} is already ' +
                   'in the group {} in the hdf5 file {}')
            msg = msg.format(name, group, f.filename)

            if group is None:
                if name in f:
                    raise RuntimeError(msg)
            else:
                if group in f:
                    if name in f[group]:
                        raise RuntimeError(msg)

        # Save the data
        util.save_dataset(f, name, data=data,
                          group=group, group_attrs=group_attrs)

    def perform_extra_consistency_checks(self):
        pass

    def perform_consistency_checks(self):
        with h5py.File(self.filename, 'r') as f:
            assert f['Header'].attrs['Time'] == self.snap.Header['Time']

        self.perform_extra_consistency_checks()

    def finalize(self):
        """
        """
        import os
        if self.mode == 'w':
            os.rename(self.tmp_filename, self.filename)
