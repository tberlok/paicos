import numpy as np
import os
import h5py

class fof_catalog:
  def __init__(self, basedir, snapnum, verbose=False):    
    if os.path.exists(basedir + "/fof_tab_" + str(snapnum).zfill(3) + ".hdf5"):
      self.multi_file = False
      self.first_file_name = basedir + "/fof_tab_" + str(snapnum).zfill(3) + ".hdf5"
    elif os.path.exists(basedir + "/groups_" + str(snapnum).zfill(3) + "/fof_tab_" + str(snapnum).zfill(3) + ".0.hdf5"):
      self.multi_file = True
      self.first_file_name = basedir + "/groups_" + str(snapnum).zfill(3) + "/fof_tab_" + str(snapnum).zfill(3) + ".0.hdf5"
      self.no_groupdir = False
    elif os.path.exists(basedir + "/fof_tab_" + str(snapnum).zfill(3) + ".0.hdf5"):
      self.multi_file = True
      self.first_file_name = basedir + "/fof_tab_" + str(snapnum).zfill(3) + ".0.hdf5"
      self.no_groupdir = True
    else:
      print("no Subfind catalog found")
      assert False

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
      if self.multi_file == False:
        cur_filename = self.first_file_name
      else:
        if self.no_groupdir==False:
          cur_filename = basedir + "/groups_" + str(snapnum).zfill(3) + "/fof_tab_" + str(snapnum).zfill(3) + "."+str(ifile)+".hdf5"
        else:
          cur_filename = basedir + "/fof_tab_" + str(snapnum).zfill(3) + "."+str(ifile)+".hdf5"

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
            self.Group[ikey] = np.empty(self.ngroups, dtype=f["Group/"+ikey].dtype) 
          elif f["Group/"+ikey].shape.__len__() == 2:
            self.Group[ikey] = np.empty((self.ngroups, f["Group/"+ikey].shape[1]), dtype=f["Group/"+ikey].dtype)
          else:
            assert False

        for ikey in f["Subhalo"].keys():
          if f["Subhalo/"+ikey].shape.__len__() == 1:
            self.Sub[ikey] = np.empty(self.nsubs, dtype=f["Subhalo/"+ikey].dtype)
          elif f["Subhalo/"+ikey].shape.__len__() == 2:
            self.Sub[ikey] = np.empty((self.nsubs, f["Subhalo/"+ikey].shape[1]), dtype=f["Subhalo/"+ikey].dtype)
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

