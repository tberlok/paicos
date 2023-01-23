import paicos as pa
import numpy as np

pa.use_units(True)

snap = pa.Snapshot(pa.root_dir + '/data', 247)
center = snap.Cat.Group['GroupPos'][0]

if pa.settings.use_units:
    r_max = 10000*center.unit_quantity
else:
    r_max = 10000

bins = np.linspace(0, r_max, 150)

radial_filename = pa.root_dir + '/data/radial_filename_247.hdf5'
radial_profile = pa.RadialProfiles(radial_filename,
                                   snap, center, r_max, bins)
radial_profile.finalize()
