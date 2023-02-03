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

            file = h5py.File(cur_filename, "r")

            ng = int(file["Header"].attrs["Ngroups_ThisFile"])

            if "Nsubgroups_ThisFile" in file["Header"].attrs.keys():
                ns = int(file["Header"].attrs["Nsubgroups_ThisFile"])
            else:
                ns = int(file["Header"].attrs["Nsubhalos_ThisFile"])

            # initialze arrays
            if ifile == 0:
                self.Group = {}
                self.Sub = {}
                for ikey in file["Group"].keys():
                    if len(file["Group/" + ikey].shape) == 1:
                        self.Group[ikey] = np.empty(
                            self.ngroups, dtype=file["Group/" + ikey].dtype)
                    elif len(file["Group/" + ikey].shape) == 2:
                        self.Group[ikey] = np.empty(
                            (self.ngroups, file["Group/" + ikey].shape[1]),
                            dtype=file["Group/" + ikey].dtype)
                    else:
                        assert False

                for ikey in file["Subhalo"].keys():
                    if len(file["Subhalo/" + ikey].shape) == 1:
                        self.Sub[ikey] = np.empty(
                            self.nsubs, dtype=file["Subhalo/" + ikey].dtype)
                    elif len(file["Subhalo/" + ikey].shape) == 2:
                        self.Sub[ikey] = np.empty(
                            (self.nsubs, file["Subhalo/" + ikey].shape[1]),
                            dtype=file["Subhalo/" + ikey].dtype)
                    else:
                        assert False

            # read group data
            for ikey in file["Group"].keys():
                self.Group[ikey][skip_gr:skip_gr + ng] = file["Group/" + ikey]

            # read subhalo data
            for ikey in file["Subhalo"].keys():
                self.Sub[ikey][skip_sub:skip_sub + ns] = file["Subhalo/" + ikey]

            skip_gr += ng
            skip_sub += ns

            file.close()

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
