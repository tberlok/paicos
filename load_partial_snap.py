import h5py
import numpy as np

# --- constants ---

from scipy.constants import parsec as pc # all in SI
from scipy.constants import m_p as mproton
from scipy.constants import m_e as melectron
from scipy.constants import k as kboltzmann
from scipy.constants import G as Gnewton
from scipy.constants import c as clight
from scipy.constants import eV
from scipy.constants import e as electron_charge
kpc = 1.0e3 * pc
Mpc = 1.0e6 * pc
keV = 1000.0*eV
msun = 1.9884430e30
mhydrogen = mproton + melectron

fhydrogen = 0.76
mmean_ionized = (1.0+(1.0-fhydrogen)/fhydrogen)/(2.0+3.0*(1.0-fhydrogen)/(4.0*fhydrogen))


class snapshot:
    """
    Brief version of Ewald's snapfile class useful
    for parallel analysis.
    Loads the partial files instead of everything
    """
    def __init__(self, output_folder, snapnum, partfile=None):
        import arepo_subf, arepo_fof

        if output_folder[-1] != '/':
            output_folder += '/'

        if partfile is None:
            filename = output_folder + 'snap_{:03d}.hdf5'.format(snapnum)
        else:
            filename = output_folder + 'snapdir_{:03d}/snap_{:03d}.{}.hdf5'.format(snapnum, snapnum, partfile)

        # get header of first file
        f = h5py.File(filename, 'r')
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

        self.filename = filename
        self.P = dict()

        # get subfind catalog
        try:
            self.Cat = arepo_subf.subfind_catalog(self.basedir, self.snapnum, verbose=self.verbose)
        except:
            self.Cat = None

        if self.Cat is None:
            try:
                self.Cat = arepo_fof.fof_catalog(self.basedir, self.snapnum, verbose=self.verbose)
            except:
                print('no fof or subfind found')

        self.z = self.Header["Redshift"]
        self.a = self.Header["Time"]

        try:
            self.omega_m = self.Header["Omega0"]
            self.omega_l = self.Header["OmegaLambda"]
            self.h = self.Header["HubbleParam"]
        except:
            print('not there')

        self.box = self.Header["BoxSize"]
        try:
            self.omega_b = self.Header["OmegaBaryon"]
        except:
            print('Tried to read OmegaBaryon but it is not in the header')
        try:
            self.u_m = self.Header["UnitMass_in_g"] / 1000.0 / self.h   # in kg
            self.u_l_c = self.Header["UnitLength_in_cm"] / 100.0 / self.h   # in comoving m
            self.u_l_p = self.Header["UnitLength_in_cm"] / 100.0 / self.h * self.a   # in physical m
            self.u_l_ckpc = self.u_l_c / kpc
            self.u_l_cMpc = self.u_l_c / Mpc
            self.u_l_pkpc = self.u_l_p / kpc
            self.u_l_pMpc = self.u_l_p / Mpc
            self.u_v = self.Header["UnitVelocity_in_cm_per_s"] / 100.0   # unit velocity in m/s
        except:
            print('Tried to read units but they are not in the header...')

    def info(self, PartType):
        PartType_str = 'PartType{}'.format(PartType)
        file = h5py.File(self.filename, 'r')
        print('keys for ' + PartType_str + ' are')
        print(file[PartType_str].keys())
        file.close()

    def load_data(self, PartType, field):
        key = '{}_'.format(PartType) + field
        if key not in self.P:
            PartType_str = 'PartType{}'.format(PartType)
            file = h5py.File(self.filename, 'r')
            # embed()
            shape = file[PartType_str][field].shape
            dtype = file[PartType_str][field].dtype
            # len_shape = len(shape)
            data = np.empty(shape, dtype=dtype)
            data = file[PartType_str][field][...]
            self.P.update({key: data})
            file.close()

    def get_volumes(self):
        if '0_Volumes' not in self.P:
            self.load_data(0, 'Masses')
            self.load_data(0, 'Density')

            self.P.update({'0_Volumes': self.P['0_Masses']/self.P['0_Density']})

    def get_temperatures(self):
        if "0_Temperatures" not in self.P:

            self.load_data(0, "InternalEnergy")

            try:
                self.load_data(0, "ElectronAbundance")
                mmean = 4.0 / (1.0 + 3.0*fhydrogen + 4.0*fhydrogen*self.P["0_ElectronAbundance"])
            except:
                mmean = mmean_ionized

            self.P["0_Temperatures"] = 2.0/3.0 * self.P["0_InternalEnergy"] * self.u_v**2 * mmean * mhydrogen / kboltzmann   # temperature in Kelvin

    def get_high_res_region(self, threshold):
        self.load_data(0, 'Masses')
        self.load_data(0, 'HighResGasMass')

        if threshold is None:
            threshold = -1
        self.P.update({'0_HighResIndex':
                      self.P['0_HighResGasMass']/self.P['0_Masses'] > threshold})
