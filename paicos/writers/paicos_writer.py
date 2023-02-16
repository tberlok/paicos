"""
Defines hdf5 file writers that can be read with a PaicosReader instance.
"""
import os
import h5py
from .. import util


class PaicosWriter:
    """
    This class writes data to self-documenting hdf5 files.

    It is the base class for the ArepoImage writer.
    """

    def __init__(self, reader_object, basedir,
                 basename="paicos_file", add_snapnum=True, mode='w'):
        """
        The constructor for the `PaicosWriter` class.

        Parameters
        ----------

        reader_object (obj): A PaicosReader object (or Snaphot or Catalog object).

        basedir (file path): The folder where the HDF5 file will be saved.

        basename (str, optional): Base name of the HDF5 file. Defaults
                                  to "paicos_file".

        add_snapnum (bool): Whether to add the snapnum to the HDF5 filename.
                            When True, the filename will be "basename_{:03d}.hdf5",
                            when False it will simply be "basename.hdf5"
                            Defaults to True.

        mode (string): The mode to open the file in, either 'w' for write mode
        or 'r' for read mode. (default: 'w')

        """

        self.reader_object = reader_object
        self.org_filename = reader_object.filename

        self.mode = mode

        snapnum = reader_object.snapnum

        if basedir[-1] != '/':
            basedir += '/'

        if not os.path.exists(basedir):
            os.makedirs(basedir)

        self.basedir = basedir
        self.basename = basename

        name = basename

        if add_snapnum:
            name += f'_{snapnum:03d}.hdf5'
        else:
            name += '.hdf5'
        self.filename = basedir + name
        self.tmp_filename = basedir + 'tmp_' + name

        if mode == 'w':
            util.copy_over_snapshot_information(self.org_filename,
                                                self.tmp_filename, 'w')
        else:
            self.perform_consistency_checks()

    def write_data(self, name, data, data_attrs={}, group=None, group_attrs={}):
        """
        Write a data set to the hdf5 file.

        Parameters
        ----------

        name (str): Name of the data to be written.

        data (obj): The data to be written (e.g. PaicosQuantity, PaicosTimeSeries,
                    Astropy Quantity, numpy array).

        data_attrs (dict): Dictionary of attributes for the data. E.g.
                           {'info': 'Here I have written some extra information
                                    about the data set'}. Defaults to an empty
                                    dictionary.

        group (str, optional): Group in the HDF5 file where the data should
                               be saved. Defaults to None in which case the
                               data set is saved at the top level of the hdf5 file).

        group_attrs (dict, optional): Dictionary of attributes for the group.
                                      Defaults to an empty dictionary.
        """

        # pylint: disable= dangerous-default-value

        if self.mode == 'w':
            filename = self.tmp_filename
        else:
            filename = self.filename

        file = h5py.File(filename, 'r+')

        if self.mode == 'a':
            msg = ('PaicosWriter is in amend mode but {} is already '
                   + 'in the group {} in the hdf5 file {}')
            msg = msg.format(name, group, file.filename)

            if group is None:
                if name in file:
                    raise RuntimeError(msg)
            else:
                if group in file:
                    if name in file[group]:
                        raise RuntimeError(msg)

        # Save the data
        util.save_dataset(file, name, data=data, data_attrs=data_attrs,
                          group=group, group_attrs=group_attrs)

    def perform_extra_consistency_checks(self):
        """
        Perform extra consistency checks. This can be overloaded by
        subclasses.
        """
        # pylint: disable=unnecessary-pass
        pass

    def perform_consistency_checks(self):
        """
        Perform consistency checks when trying to amend a file (to avoid
        saving data at different times, for instance)
        """
        with h5py.File(self.filename, 'r') as f:
            org_time = self.reader_object.Header['Time']
            assert f['Header'].attrs['Time'] == org_time

        self.perform_extra_consistency_checks()

    def finalize(self):
        """
        Move from a temporary to the final filename.
        """
        if self.mode == 'w':
            os.rename(self.tmp_filename, self.filename)


class PaicosTimeSeriesWriter(PaicosWriter):
    """
    Similar to the standard PaicosWriter but here we ensure that
    the snapnum is not part of the resulting filename for the hdf5 file.
    """

    def __init__(self, reader_object, basedir,
                 basename="paicos_time_series", add_snapnum=False, mode='w'):

        super().__init__(reader_object, basedir,
                         basename=basename,
                         add_snapnum=add_snapnum,
                         mode=mode)
