import numpy as np
import os
import h5py


class Catalog:
    """
    Catalog reader, originally written by Ewald Puchwein.

    The Catalog class is a class for reading subhalo or halo catalogs in Arepo
    snapshots, which are simulations of the evolution of the universe. The
    class takes in the path of the directory containing the catalog, the
    snapshot number, and a variety of other optional parameters such as
    whether to use the verbose mode, read subfind catalog or fof catalog, and
    whether to give units. The class uses this information to locate and open
    the catalog files, and loads the catalog's header, parameters, and
    configuration. The class also includes methods to extract the redshift,
    scale factor, and other properties of the catalog and the number of groups
    and subgroups in the catalog.
    """

    def __init__(self, basedir, snapnum, verbose=False, subfind_catalog=True,
                 converter=None, give_units=False):

        if subfind_catalog:
            single_file = "/fof_subhalo_tab_{:03d}.hdf5"
            multi_file = "/groups_{:03d}/fof_subhalo_tab_{:03d}.{}.hdf5"
            multi_wo_dir = "/fof_subhalo_tab_{:03d}.{}.hdf5"
        else:
            single_file = "/fof_tab_{:03d}.hdf5"
            multi_file = "/groups_{:03d}/fof_tab_{:03d}.{}.hdf5"
            multi_wo_dir = "/fof_tab_{:03d}.{}.hdf5"

        single_file = basedir + single_file.format(snapnum)
        multi_file = basedir + multi_file.format(snapnum, snapnum, '{}')
        multi_wo_dir = basedir + multi_wo_dir.format(snapnum, '{}')

        if os.path.exists(single_file):
            self.multi_file = False
            self.first_file_name = single_file
        elif os.path.exists(multi_file.format(0)):
            self.multi_file = True
            self.first_file_name = multi_file.format(0)
            self.no_groupdir = False
        elif os.path.exists(multi_wo_dir.format(0)):
            self.multi_file = True
            self.first_file_name = multi_wo_dir.format(0)
            self.no_groupdir = True
        else:
            if subfind_catalog:
                err_msg = "no subfind catalog for snapnum {} in basedir: {}"
            else:
                err_msg = "no fof catalog for snapnum {} in basedir: {}"
            raise FileNotFoundError(err_msg.format(snapnum, basedir))

        f = h5py.File(self.first_file_name, "r")
        header_attrs = f['/Header'].attrs

        self.Header = dict()
        for ikey in header_attrs.keys():
            self.Header[ikey] = header_attrs[ikey]

        f.close()

        # give names to specific header fields
        self.nfiles = self.Header["NumFiles"]
        self.ngroups = self.Header["Ngroups_Total"]

        if "Nsubgroups_Total" in self.Header.keys():
            self.nsubs = self.Header["Nsubgroups_Total"]
        else:
            self.nsubs = self.Header["Nsubhalos_Total"]

        self.z = self.Header["Redshift"]
        self.a = self.Header["Time"]

        if verbose:
            print("done reading catalog!")
            print("ngroups =", self.ngroups)
            print("nsubs =", self.nsubs)

        skip_gr = 0
        skip_sub = 0
        for ifile in range(self.nfiles):
            if self.multi_file is False:
                cur_filename = self.first_file_name
            else:
                if self.no_groupdir:
                    cur_filename = multi_wo_dir.format(ifile)
                else:
                    cur_filename = multi_file.format(ifile)

            if verbose:
                print("reading file", cur_filename)

            f = h5py.File(cur_filename, "r")

            ng = int(f["Header"].attrs["Ngroups_ThisFile"])

            if "Nsubgroups_ThisFile" in f["Header"].attrs.keys():
                ns = int(f["Header"].attrs["Nsubgroups_ThisFile"])
            else:
                ns = int(f["Header"].attrs["Nsubhalos_ThisFile"])

            # initialze arrays
            if ifile == 0:
                self.Group = dict()
                self.Sub = dict()

                for ikey in f["Group"].keys():
                    if f["Group/"+ikey].shape.__len__() == 1:
                        self.Group[ikey] = np.empty(
                            self.ngroups, dtype=f["Group/"+ikey].dtype)
                    elif f["Group/"+ikey].shape.__len__() == 2:
                        self.Group[ikey] = np.empty(
                            (self.ngroups, f["Group/"+ikey].shape[1]),
                            dtype=f["Group/"+ikey].dtype)
                    else:
                        assert False

                for ikey in f["Subhalo"].keys():
                    if f["Subhalo/"+ikey].shape.__len__() == 1:
                        self.Sub[ikey] = np.empty(
                            self.nsubs, dtype=f["Subhalo/"+ikey].dtype)
                    elif f["Subhalo/"+ikey].shape.__len__() == 2:
                        self.Sub[ikey] = np.empty(
                            (self.nsubs, f["Subhalo/"+ikey].shape[1]),
                            dtype=f["Subhalo/"+ikey].dtype)
                    else:
                        assert False

            # read group data
            for ikey in f["Group"].keys():
                self.Group[ikey][skip_gr:skip_gr+ng] = f["Group/"+ikey]

            # read subhalo data
            for ikey in f["Subhalo"].keys():
                self.Sub[ikey][skip_sub:skip_sub+ns] = f["Subhalo/"+ikey]

            skip_gr += ng
            skip_sub += ns

            f.close()

        from . import settings
        give_units = settings.use_units or give_units

        if give_units and converter is not None:

            mass_keys = ['GroupMass',
                         'Group_M_Crit200',
                         'Group_M_Crit500',
                         'Group_M_Mean200',
                         'Group_M_TopHat200']

            pos_keys = ['GroupPos',
                        'Group_R_Crit200',
                        'Group_R_Crit500',
                        'Group_R_Mean200',
                        'Group_R_TopHat200']

            for key in pos_keys:
                self.Group[key] = converter.get_paicos_quantity(
                                    self.Group[key], 'Coordinates')

            for key in mass_keys:
                self.Group[key] = converter.get_paicos_quantity(
                                     self.Group[key], 'Masses')
