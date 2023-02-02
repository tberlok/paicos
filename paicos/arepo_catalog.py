import numpy as np
import h5py
from .paicos_readers import PaicosReader
from . import settings


class Catalog(PaicosReader):
    def __init__(self, basedir='.', snapnum=None,
                 to_physical=False, subfind_catalog=True, verbose=False):

        if subfind_catalog:
            basename = 'fof_subhalo_tab'
        else:
            basename = 'fof_tab'

        super().__init__(basedir=basedir, snapnum=snapnum, basename=basename,
                         load_all=False, to_physical=to_physical,
                         basesubdir='groups', verbose=verbose)

        # give names to specific header fields
        self.nfiles = self.Header["NumFiles"]
        self.ngroups = self.Header["Ngroups_Total"]

        if "Nsubgroups_Total" in self.Header.keys():
            self.nsubs = self.Header["Nsubgroups_Total"]
        else:
            self.nsubs = self.Header["Nsubhalos_Total"]

        # Load all data
        self.load_all_data()

    def load_data(self):
        """
        Overwrite the base class method
        """
        pass

    def load_all_data(self):

        skip_gr = 0
        skip_sub = 0
        for ifile in range(self.nfiles):
            if self.multi_file is False:
                cur_filename = self.filename
            else:
                if self.no_subdir:
                    cur_filename = self.multi_wo_dir.format(ifile)
                else:
                    cur_filename = self.multi_filename.format(ifile)

            if self.verbose:
                print("reading file", cur_filename)

            f = h5py.File(cur_filename, "r")

            ng = int(f["Header"].attrs["Ngroups_ThisFile"])

            if "Nsubgroups_ThisFile" in f["Header"].attrs.keys():
                ns = int(f["Header"].attrs["Nsubgroups_ThisFile"])
            else:
                ns = int(f["Header"].attrs["Nsubhalos_ThisFile"])

            # initialze arrays
            if ifile == 0:
                self.Group = {}
                self.Sub = {}
                for ikey in f["Group"].keys():
                    if f["Group/" + ikey].shape.__len__() == 1:
                        self.Group[ikey] = np.empty(
                            self.ngroups, dtype=f["Group/" + ikey].dtype)
                    elif f["Group/" + ikey].shape.__len__() == 2:
                        self.Group[ikey] = np.empty(
                            (self.ngroups, f["Group/" + ikey].shape[1]),
                            dtype=f["Group/" + ikey].dtype)
                    else:
                        assert False

                for ikey in f["Subhalo"].keys():
                    if f["Subhalo/" + ikey].shape.__len__() == 1:
                        self.Sub[ikey] = np.empty(
                            self.nsubs, dtype=f["Subhalo/" + ikey].dtype)
                    elif f["Subhalo/" + ikey].shape.__len__() == 2:
                        self.Sub[ikey] = np.empty(
                            (self.nsubs, f["Subhalo/" + ikey].shape[1]),
                            dtype=f["Subhalo/" + ikey].dtype)
                    else:
                        assert False

            # read group data
            for ikey in f["Group"].keys():
                self.Group[ikey][skip_gr:skip_gr + ng] = f["Group/" + ikey]

            # read subhalo data
            for ikey in f["Subhalo"].keys():
                self.Sub[ikey][skip_sub:skip_sub + ns] = f["Subhalo/" + ikey]

            skip_gr += ng
            skip_sub += ns

            f.close()

        if settings.use_units:
            for key in list(self.Group.keys()):
                self.Group[key] = self.get_paicos_quantity(
                    self.Group[key], key,
                    field='groups')
                if not hasattr(self.Group[key], 'unit'):
                    del self.Group[key]

            for key in list(self.Sub.keys()):
                self.Sub[key] = self.get_paicos_quantity(
                    self.Sub[key], key,
                    field='subhalos')
                if not hasattr(self.Sub[key], 'unit'):
                    del self.Sub[key]
