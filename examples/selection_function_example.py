import numpy as np
import paicos as pa
pa.use_units(True)

snap = pa.Snapshot(pa.root_dir + '/data', 247)
center = snap.Cat.Group['GroupPos'][0]
r_max = 3000*snap.converter.length

# Calculate index in the "slow" but simple way
r = np.sqrt(np.sum((snap['0_Coordinates']-center[None, :])**2., axis=1))
index_slow = r < r_max

# Use OpenMP parallel Cython code
index = pa.util.get_index_of_radial_range(snap['0_Coordinates'],
                                          center, 0., r_max)

# Check that we get the same result
np.testing.assert_array_equal(index_slow, index)
