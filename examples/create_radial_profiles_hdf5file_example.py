import paicos as pa
import numpy as np

# Open an Arepo snapshot
snap = pa.Snapshot(pa.root_dir + "/data", 247)

# Create a Paicos writer object
radfile = pa.PaicosWriter(snap, pa.root_dir + 'test_data', basename='radial')

# The center of the most massive Friends-of-friends group in the simulation
center = snap.Cat.Group["GroupPos"][0]

R200c = snap.Cat.Group['Group_R_Crit200'][0]

# The maximum radius to be considered
r_max = 10000 * center.uq

# Use OpenMP parallel Cython code to find this central sphere
index = pa.util.get_index_of_radial_range(snap['0_Coordinates'],
                                          center, 0., r_max)

# Create a new snap object which only contains the index above
snap = snap.select(index)


# Calculate the radial distances (a bit duplicate here...)
r = np.sqrt(np.sum((snap["0_Coordinates"] - center[None, :]) ** 2.0,
            axis=1))

# Set up the binning
bins = [1e-2 * r_max, r_max, 100]

# Create a histogram object
h_r = pa.Histogram(r, bins=bins, logscale=True)

# Save the radius
radfile.write_data('bin_centers', h_r.bin_centers)

# Bin volumes (of the shells)
bin_volumes = np.diff(4 / 3 * np.pi * h_r.bin_edges**3)
radfile.write_data('bin_volumes', bin_volumes)

gas_keys = ['0_Masses',
            '0_Volume',
            '0_TemperaturesTimesMasses',
            '0_MagneticFieldSquaredTimesVolume',
            '0_PressureTimesVolume']

# Do the histograms and save at the same time
for key in gas_keys:
    radfile.write_data(key, h_r.hist(snap[key]))

# It will probably also be useful to have some group properties
# We save the 10 most massive FOF groups (sorted according to their M200_crit)
index = np.argsort(snap.Cat.Group['Group_M_Crit200'])[::-1]
for key in snap.Cat.Group.keys():
    radfile.write_data(key, snap.Cat.Group[key][index[:10]], group='Group')

# Short hand access to the most massive will probably be nice
radfile.write_data('R200c', R200c)
radfile.write_data('center', center)

# Let us now also add some other parttype profiles

for parttype in range(1, snap.nspecies):
    pstr = f'{parttype}_'
    # Re-open the Arepo snapshot
    snap = pa.Snapshot(pa.root_dir + "/data", 247)

    # Use OpenMP parallel Cython code to find this central sphere
    index = pa.util.get_index_of_radial_range(snap[pstr + 'Coordinates'],
                                              center, 0., r_max)

    # Create a new snap object which only contains the index above
    snap = snap.select(index)

    r = np.sqrt(np.sum((snap[pstr + "Coordinates"] - center[None, :]) ** 2.0,
                axis=1))

    # Create a new histogram object
    h_r = pa.Histogram(r, bins=bins, logscale=True)

    # Write the mass profile
    radfile.write_data(pstr + 'Masses', h_r.hist(snap[pstr + 'Masses']))

# Rename from a tmp_*.hdf5 file to the final filename
radfile.finalize()

print(radfile.filename)
