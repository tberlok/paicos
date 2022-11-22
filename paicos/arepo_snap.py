# --- import Python libs ---

from .arepo_catalog import Catalog
import numpy as np
import os
import time
import h5py

# --- constants ---

from scipy.constants import parsec as pc  # all in SI
from scipy.constants import m_p as mproton
from scipy.constants import m_e as melectron
from scipy.constants import k as kboltzmann
from scipy.constants import eV
kpc = 1.0e3 * pc
Mpc = 1.0e6 * pc
keV = 1000.0*eV
msun = 1.9884430e30
mhydrogen = mproton + melectron


class Snapshot:
    """
    A class for reading in Arepo snapshots.
    Based on script written by Ewald Puchwein
    """
    def __init__(self, basedir, snapnum, snap_basename="snap", verbose=False,
                 no_snapdir=False, load_catalog=True):
        self.basedir = basedir
        self.snapnum = snapnum
        self.snap_basename = snap_basename
        self.verbose = verbose
        self.no_snapdir = no_snapdir

        # in case single file
        self.snapname = self.basedir + "/" + \
            snap_basename + "_" + str(self.snapnum).zfill(3)
        self.multi_file = False
        self.first_snapfile_name = self.snapname + ".hdf5"

        # if multiple files
        if not os.path.exists(self.first_snapfile_name):
            if not self.no_snapdir:
                self.snapname = self.basedir + "/" + "snapdir_" + \
                    str(self.snapnum).zfill(3) + "/" + \
                    snap_basename + "_" + str(self.snapnum).zfill(3)
            else:
                self.snapname = self.basedir + "/" + \
                    snap_basename + "_" + str(self.snapnum).zfill(3)
            self.first_snapfile_name = self.snapname+".0.hdf5"
            assert os.path.exists(self.first_snapfile_name)
            self.multi_file = True

        if self.verbose:
            print("snapshot", snapnum, "found")

        # get header of first file
        f = h5py.File(self.first_snapfile_name, 'r')
        header_attrs = f['/Header'].attrs

        self.Header = dict()
        for ikey in header_attrs.keys():
            self.Header[ikey] = header_attrs[ikey]

        self.have_params = False
        if "Parameters" in f.keys():
            parameters_attrs = f['/Parameters'].attrs
            self.Parameters = dict()

            for ikey in parameters_attrs.keys():
                self.Parameters[ikey] = parameters_attrs[ikey]

            self.have_params = True

        f.close()

        self.nfiles = self.Header["NumFilesPerSnapshot"]
        self.npart = self.Header["NumPart_Total"]
        self.nspecies = self.npart.size
        self.masstable = self.Header["MassTable"]

        if self.verbose:
            print("has", self.nspecies, "particle types")
            print("with npart =", self.npart)

        self.z = self.Header["Redshift"]
        self.a = self.Header["Time"]

        if self.verbose:
            print("at z =", self.z)

        self.box = self.Header["BoxSize"]

        if self.have_params:
            pardict = self.Parameters
        else:
            pardict = self.Header

        self.omega_m = pardict["Omega0"]
        self.omega_l = pardict["OmegaLambda"]
        self.omega_b = pardict["OmegaBaryon"]
        self.h = pardict["HubbleParam"]

        self.u_m = pardict["UnitMass_in_g"] / 1000.0 / self.h   # in kg
        self.u_l_c = pardict["UnitLength_in_cm"] / \
            100.0 / self.h   # in comoving m
        self.u_l_p = pardict["UnitLength_in_cm"] / \
            100.0 / self.h * self.a   # in physical m
        self.u_l_ckpc = self.u_l_c / kpc
        self.u_l_cMpc = self.u_l_c / Mpc
        self.u_l_pkpc = self.u_l_p / kpc
        self.u_l_pMpc = self.u_l_p / Mpc
        self.u_v = pardict["UnitVelocity_in_cm_per_s"] / \
            100.0   # unit velocity in m/s

        # get subfind catalog
        if load_catalog:
            try:
                self.Cat = Catalog(
                    self.basedir, self.snapnum, verbose=self.verbose,
                    subfind_catalog=True)
            except FileNotFoundError:
                self.Cat = None

            # If no subfind catalog found, then try for a fof catalog
            if self.Cat is None:
                try:
                    self.Cat = Catalog(
                        self.basedir, self.snapnum, verbose=self.verbose,
                        subfind_catalog=False)
                except FileNotFoundError:
                    import warnings
                    warnings.warn('no catalog found', FileNotFoundError)

        self.P = dict()   # particle data
        self.P_attrs = dict()  # attributes

    def info(self, PartType, verbose=True):
        PartType_str = 'PartType{}'.format(PartType)
        with h5py.File(self.first_snapfile_name, 'r') as file:
            if PartType_str in list(file.keys()):
                keys = list(file[PartType_str].keys())
                if verbose:
                    print('keys for ' + PartType_str + ' are')
                    print(keys)
                return keys
            else:
                if verbose:
                    print('PartType not in hdf5 file')
                return None

    def load_data(self, particle_type, blockname):
        assert particle_type < self.nspecies

        P_key = str(particle_type)+"_"+blockname
        datname = "PartType"+str(particle_type)+"/"+blockname
        PartType_str = 'PartType{}'.format(particle_type)
        if P_key in self.P:
            if self.verbose:
                print(blockname, "for species",
                      particle_type, "already in memory")
            return
        elif self.verbose:
            print("loading block", blockname,
                  "for species", particle_type, "...")
            start_time = time.time()

        if blockname == "Masses" and self.masstable[particle_type] > 0:
            self.P[P_key] = self.masstable[particle_type] * \
                np.ones(self.npart[particle_type], dtype=np.float32)
            if self.verbose:
                print("... got value from MassTable in header!")
            return

        skip_part = 0

        for ifile in range(self.nfiles):
            cur_filename = self.snapname

            if self.multi_file:
                cur_filename += "." + str(ifile)

            cur_filename += ".hdf5"

            f = h5py.File(cur_filename, "r")

            np_file = f["Header"].attrs["NumPart_ThisFile"][particle_type]

            if ifile == 0:   # initialize array
                if f[datname].shape.__len__() == 1:
                    self.P[P_key] = np.empty(
                        self.npart[particle_type], dtype=f[datname].dtype)
                else:
                    self.P[P_key] = np.empty(
                        (self.npart[particle_type], f[datname].shape[1]),
                        dtype=f[datname].dtype)
                # Load attributes
                data_attributes = dict(f[PartType_str][blockname].attrs)
                if len(data_attributes) > 0:
                    data_attributes.update({'small_h': self.h,
                                            'scale_factor': self.a})
                self.P_attrs[P_key] = data_attributes

            self.P[P_key][skip_part:skip_part+np_file] = f[datname]

            skip_part += np_file

        if self.verbose:
            print("... done! (took", time.time()-start_time, "s)")

    def remove_data(self, particle_type, blockname):
        """
        Remove data from object. Sometimes useful for for large datasets
        """
        P_key = str(particle_type)+"_"+blockname
        if P_key in self.P:
            del self.P[P_key]
        if P_key in self.P_attrs:
            del self.P_attrs[P_key]

    def get_high_res_region(self, threshold=0.9):
        if "0_HighResIndex" in self.P:
            return

        if self.verbose:
            print("computing indices of HighResGasMass")

        self.load_data(0, "HighResGasMass")
        self.load_data(0, "Masses")

        self.P["0_HighResIndex"] = self.P['0_HighResGasMass'] > threshold * \
            self.P['0_Masses']

    def get_volumes(self):
        if "0_Volumes" in self.P:
            return

        if self.verbose:
            print("computing volumes ...")
            start_time = time.time()

        self.load_data(0, "Masses")
        self.load_data(0, "Density")
        self.P["0_Volumes"] = self.P["0_Masses"] / self.P["0_Density"]

        if self.verbose:
            print("... done! (took", time.time()-start_time, "s)")

    def get_temperatures(self):
        if "0_Temperatures" in self.P:
            return

        if self.verbose:
            print("computing temperatures ...")
            start_time = time.time()

        self.load_data(0, "InternalEnergy")

        fhydrogen = 0.76

        if "ElectronAbundance" in self.info(0):
            self.load_data(0, "ElectronAbundance")
            mmean = 4.0 / (1.0 + 3.0*fhydrogen + 4.0 *
                           fhydrogen*self.P["0_ElectronAbundance"])
        else:
            mmean_ionized = (1.0+(1.0-fhydrogen)/fhydrogen) / \
                (2.0+3.0*(1.0-fhydrogen)/(4.0*fhydrogen))
            ne_ionized = 1.0 + 2.0*(1.0-fhydrogen)/(4.0*fhydrogen)
            mmean = mmean_ionized

        self.P["0_Temperatures"] = 2.0/3.0 * self.P["0_InternalEnergy"] * \
            self.u_v**2 * mmean * mhydrogen / kboltzmann   # temperature in Kelvin

        if self.verbose:
            print("... done! (took", time.time()-start_time, "s)")

    # find subhalos that particles belong to
    def get_host_subhalos(self, particle_type):
        if str(particle_type)+"_HostSub" in self.P:
            return

        if self.verbose:
            print("getting host subhalos ...")
            start_time = time.time()

        # -2 if not part of any FoF group
        hostsubhalos = -2*np.ones(self.npart[particle_type], dtype=np.int32)
        hostsubhalos[0:(self.Cat.Group["GroupLenType"][:, particle_type]).sum(
            dtype=np.int64)] = -1   # -1 if not part of any subhalo

        firstpart_gr = 0
        cur_sub = 0
        for igr in range(self.Cat.ngroups):
            firstpart_sub = firstpart_gr

            for isub in range(self.Cat.Group["GroupNsubs"][igr]):
                hostsubhalos[firstpart_sub:firstpart_sub +
                             self.Cat.Sub["SubhaloLenType"][cur_sub, particle_type]] = cur_sub

                firstpart_sub += self.Cat.Sub["SubhaloLenType"][cur_sub,
                                                                particle_type]
                cur_sub += 1

            firstpart_gr += self.Cat.Group["GroupLenType"][igr, particle_type]

        self.P[str(particle_type)+"_HostSub"] = hostsubhalos

        if self.verbose:
            print("... done! (took", time.time()-start_time, "s)")
