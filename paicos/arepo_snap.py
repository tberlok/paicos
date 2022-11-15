# --- import Python libs ---

import numpy as np
import os
import time
import h5py

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
ne_ionized = 1.0 + 2.0*(1.0-fhydrogen)/(4.0*fhydrogen)

# --- import own libs ---

from . import arepo_subf
from . import arepo_fof

# --- helper functions ---


def box_wrap_diff(pos, boxsize): #wraps to -0.5*boxsize - 0.5*boxsize
  ind = np.where(pos < -0.5*boxsize)
  pos[ind] += boxsize
  
  ind = np.where(pos > 0.5*boxsize)
  pos[ind] -= boxsize
  
  assert (pos >= -0.5*boxsize).all()
  assert (pos <= 0.5*boxsize).all()


# --- snapshot class ---


class snapshot:
  def __init__(self, basedir, snapnum, snap_basename="snap", verbose=False, no_snapdir=False):
    self.basedir = basedir
    self.snapnum = snapnum
    self.snap_basename = snap_basename
    self.verbose = verbose
    self.no_snapdir = no_snapdir

    # in case single file
    self.snapname = self.basedir + "/" + snap_basename + "_" + str(self.snapnum).zfill(3)
    self.multi_file = False
    self.first_snapfile_name = self.snapname + ".hdf5"

    # if multiple files    
    if not os.path.exists(self.first_snapfile_name):
      if self.no_snapdir==False:
        self.snapname = self.basedir + "/" + "snapdir_" + str(self.snapnum).zfill(3) + "/" + snap_basename + "_" + str(self.snapnum).zfill(3)
      else:
        self.snapname = self.basedir + "/" + snap_basename + "_" + str(self.snapnum).zfill(3)
      self.first_snapfile_name = self.snapname+".0.hdf5"
      assert os.path.exists(self.first_snapfile_name)
      self.multi_file = True
    
    if self.verbose:
      print("snapshot", snapnum, "found")

    # get header of first file
    f = h5py.File(self.first_snapfile_name,'r')
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
    self.u_l_c = pardict["UnitLength_in_cm"] / 100.0 / self.h   # in comoving m
    self.u_l_p = pardict["UnitLength_in_cm"] / 100.0 / self.h * self.a   # in physical m   
    self.u_l_ckpc = self.u_l_c / kpc
    self.u_l_cMpc = self.u_l_c / Mpc
    self.u_l_pkpc = self.u_l_p / kpc
    self.u_l_pMpc = self.u_l_p / Mpc
    self.u_v = pardict["UnitVelocity_in_cm_per_s"] / 100.0   # unit velocity in m/s
    
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

    self.P = dict()   # particle data

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

  def get_high_res_region(self, threshold=0.9):
    if "0_HighResIndex" in self.P:
      return

    if self.verbose:
      print("computing indices of HighResGasMass")

    self.load_data(0, "HighResGasMass")
    self.load_data(0, "Masses")

    self.P["0_HighResIndex"] = self.P['0_HighResGasMass'] > threshold * self.P['0_Masses']

  def load_data(self, particle_type, blockname):
    assert particle_type < self.nspecies

    if str(particle_type)+"_"+blockname in self.P:
      if self.verbose:
        print(blockname, "for species", particle_type, "already in memory")
      return
    elif self.verbose:
      print("loading block", blockname, "for species", particle_type, "...")
      start_time = time.time()

    datname = "PartType"+str(particle_type)+"/"+blockname

    if blockname == "Masses" and self.masstable[particle_type] > 0:
      self.P[str(particle_type)+"_"+blockname] = self.masstable[particle_type] * np.ones(self.npart[particle_type], dtype=np.float32)
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
          self.P[str(particle_type)+"_"+blockname] = np.empty(self.npart[particle_type], dtype = f[datname].dtype)
        else:
          self.P[str(particle_type)+"_"+blockname] = np.empty((self.npart[particle_type], f[datname].shape[1]), dtype = f[datname].dtype)

      self.P[str(particle_type)+"_"+blockname][skip_part:skip_part+np_file] = f[datname]

      skip_part += np_file
    
    if blockname=="Velocities":
      self.P[str(particle_type)+"_"+blockname] *= np.sqrt(self.a)   # convert to physical velocity
    
    if self.verbose:
      print("... done! (took", time.time()-start_time, "s)")


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

    try:
      self.load_data(0, "ElectronAbundance")
      mmean = 4.0 / (1.0 + 3.0*fhydrogen + 4.0*fhydrogen*self.P["0_ElectronAbundance"])
    except:
      mmean = mmean_ionized

    self.P["0_Temperatures"] = 2.0/3.0 * self.P["0_InternalEnergy"] * self.u_v**2 * mmean * mhydrogen / kboltzmann   # temperature in Kelvin

    if self.verbose:
      print("... done! (took", time.time()-start_time, "s)")


  def get_host_subhalos(self, particle_type):   # find subhalos that particles belong to
    if str(particle_type)+"_HostSub" in self.P:
      return

    if self.verbose:
      print("getting host subhalos ...")
      start_time = time.time()

    hostsubhalos = -2*np.ones(self.npart[particle_type], dtype=np.int32)                      # -2 if not part of any FoF group
    hostsubhalos[0:(self.Cat.Group["GroupLenType"][:,particle_type]).sum(dtype=np.int64)] = -1   # -1 if not part of any subhalo

    firstpart_gr = 0
    cur_sub = 0
    for igr in range(self.Cat.ngroups):
      firstpart_sub = firstpart_gr

      for isub in range(self.Cat.Group["GroupNsubs"][igr]):
        hostsubhalos[firstpart_sub:firstpart_sub+self.Cat.Sub["SubhaloLenType"][cur_sub,particle_type]] = cur_sub

        firstpart_sub += self.Cat.Sub["SubhaloLenType"][cur_sub,particle_type]
        cur_sub += 1

      firstpart_gr += self.Cat.Group["GroupLenType"][igr,particle_type]

    self.P[str(particle_type)+"_HostSub"] = hostsubhalos

    if self.verbose:
      print("... done! (took", time.time()-start_time, "s)")


  def get_mass_density_cummass_profile_species(self, sub, species, rmin, rmax, nbins=500, bin_scaling="log"):
    self.load_data(species, "Coordinates")
    self.load_data(species, "Masses")
    sub_pos = self.Cat.Sub["SubhaloPos"][sub]

    pos = self.P[str(species)+"_Coordinates"].copy()
    pos -= sub_pos
    box_wrap_diff(pos, self.box)
    rad = np.sqrt(np.sum(pos**2, axis=1))
    del pos

    if bin_scaling == "log":
      bin_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    elif bin_sacaling == "lin":
      bin_edges = np.linspace(rmin, rmax, nbins+1)
    else:
      assert False    

    volumes = 4.0*np.pi/3.0*(bin_edges[1:]**3-bin_edges[:-1]**3)
    
    mass_profile,buff = np.histogram(rad, bins=bin_edges, weights=np.float64(self.P[str(species)+"_Masses"]))
    density_profile = mass_profile / volumes
    cummass_profile = mass_profile.cumsum()

    r_center = bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
    r_outer = bin_edges[1:]

    return r_center, mass_profile, density_profile, r_outer, cummass_profile


  def get_gas_slice(self, xc, yc, zc, sidelength, npix=512, nthreads=4, rot_mat=None):
    from get_slice import get_slice
    
    if self.verbose:
      print("getting gas slice ...")
      start_time = time.time()
    
    a_z = 0.0
    a_x = 0.0
    a_z2 = 0.0
    
    #mapind = get_slice(self.P["0_Coordinates"][:,0], self.P["0_Coordinates"][:,1], self.P["0_Coordinates"][:,2], self.npart[0], self.box, zc, xc, yc, sidelength, a_z, a_x, a_z2, npix, 0, 0)
    mapind = get_slice(np.float32(self.P["0_Coordinates"][:,0]), np.float32(self.P["0_Coordinates"][:,1]), np.float32(self.P["0_Coordinates"][:,2]), self.npart[0], self.box, zc, xc, yc, sidelength, a_z, a_x, a_z2, npix, 0, 0)
    
    if self.verbose:
      print("... done! (took", time.time()-start_time, "s)")
      
    return mapind
    

  def get_gas_projection(self, xc, yc, zc, sidelength, thickness, npix=512, nthreads=4, rot_mat=None):
    from put_grid import put_grid
    
    if self.verbose:
      print("getting gas projection ...")

    self.get_volumes()

    periodic = False
    if sidelength == self.box:
      periodic = True 

    self.load_data(0, "Coordinates")
    pos = self.P["0_Coordinates"].copy()
    pos[:,0] -= xc
    pos[:,1] -= yc
    pos[:,2] -= zc

    box_wrap_diff(pos, self.box)

    # could rotate here
    if rot_mat is not None:
      assert((np.dot(rot_mat.transpose(),rot_mat) - np.identity(3)).max() < 1.0e-10)
      pos = np.dot(rot_mat,pos.transpose()).transpose()

    ind = np.where((pos[:,2] > -0.5*thickness) & (pos[:,2] < 0.5*thickness))[0]

    nvol = 64.0
    #hsml = (nvol*(self.P["0_Volumes"][ind])/(4.0*np.pi/3.0))**(1.0/3.0)
    hsml = np.cbrt(nvol*(self.P["0_Volumes"][ind])/(4.0*np.pi/3.0))   # much faster but reuires new numpy
    
    if self.verbose:
      print("calling put_grid")

    gasmap = put_grid(npix, 0, 0, sidelength, pos[ind,0] , pos[ind,1], hsml, self.P["0_Masses"][ind], periodic, num_threads=nthreads)

    if self.verbose:
      print("... done!")

    return gasmap


  def get_gas_projection_sub(self, sub, sidelength, thickness, npix=512, nthreads=4, rot=None, rot_rmax_pkpc=10.0, rot_pt=0):    
    subpos = self.Cat.Sub["SubhaloPos"][sub]

    rot_mat = None
    if rot:
      L = self.get_angular_momentum_sub(sub, rot_rmax_pkpc/self.u_l_pkpc, particle_type=rot_pt)
      assert rot in ["face-on","edge-on"]
      rot_mat = rot_mat_from_2vecs(L, [1.0,0.0,0.0], rot)

    return self.get_gas_projection(subpos[0], subpos[1], subpos[2], sidelength, thickness, npix=npix, nthreads=nthreads, rot_mat=rot_mat)


  def get_angular_momentum_sub(self, sub, rmax, particle_type):
    rmax2 = rmax**2
    
    self.load_data(particle_type, "Coordinates")
    self.get_host_subhalo(particle_type)

    inds =  np.where(self.P[str(particle_type)+"_HostSub"] == sub)[0]
    pos = self.P[str(particle_type)+"_Coordinates"][inds]

    subpos = self.Cat.Sub["SubhaloPos"][sub]
    pos = pos - subpos
    box_wrap_diff(pos, self.box)
    
    indr = np.where((pos**2).sum(axis=1) < rmax2)[0]
    pos = pos[indr] * self.a   # now in physical units

    self.load_data(particle_type, "Masses")
    self.load_data(particle_type, "Velocities")
    
    mass = self.P[str(particle_type)+"_Masses"][inds[indr]]
    vel = self.P[str(particle_type)+"_Velocities"][inds[indr]]
 
    L_ang_mom = np.sum(mass[:,None] * np.cross(pos,vel), dtype=np.float64, axis=0)

    return L_ang_mom

  
  #def build_tree(self,

 
  #def shoot_los(self, p1, p2, npix):
  #  los_pos = (p2 - p1)/npix * (arange(npix, dtype=float64).reshape(npix,1) + 0.5)
    
