import paicos as pa
from paicos import root_dir

"""
Here we show how to efficiently calculate center-of-mass and
total angular momentum using snapshot methods
"""

snap = pa.Snapshot(root_dir + '/data', 247)

group_pos = snap.Cat.Group['GroupPos'][0]
R200c = snap.Cat.Group['Group_R_Crit200'][0]

# Create a new snapshot which only contains stuff inside 3 R200c from the catalog
snap = snap.radial_select(group_pos, 3 * R200c)

# Center of mass for the gas (which is parttype 0)
center_gas = snap.center_of_mass(parttype=0)

# Total center of mass
center_tot, _ = snap.center_of_mass()

# Total gas angular momentum
gas_angular_momentum = snap.total_angular_momentum(center_gas, parttype=0)

# Total angular momentum
tot_angular_momentum, _ = snap.total_angular_momentum(center_tot)
