import paicos as pa
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Read a Paicos test data set without units
# store its values as simple numpy arrays and
# then delete the pa.Snapshot object
pa.use_units(False)

s = pa.Snapshot(pa.data_dir, 247, basename='reduced_snap',
                load_catalog=False)
center = [398968.4, 211682.6, 629969.9]

vol = s['0_Volume']
pos = s['0_Coordinates']
mass = s['0_Masses']

boxsize_in_code_units = s.box

del s

# Now we try the GenericSnapshot, using just the
# numpy arrays from above

pa.use_units(True)

if pa.settings.use_units:

    # Initialize a GenericSnapshot and save it
    snap = pa.GenericSnapshot(only_init=True)
    snap.give_info(boxsize_in_code_units=boxsize_in_code_units,
                   time_in_code_units=1,
                   snapnum=0,
                   length_unit=u.Unit('AU'), time_unit=u.Unit('Myr'),
                   mass_unit=u.Unit('Msun'))

    snap.set_volumes(vol)
    snap.set_positions(pos)
    snap.set_data(mass, '0_Masses', 'Msun')

    writer = pa.PaicosWriter(snap, basedir=f"{pa.data_dir}test_data", basename='paicos_file')
    for key in snap:
        writer.write_data(key, snap[key])
    writer.finalize()
    # Delete then read previously saved snapshot
    del snap
    snap = pa.GenericSnapshot(f"{pa.data_dir}test_data/paicos_file_000.hdf5")

else:
    snap = pa.GenericSnapshot(only_init=True)
    snap.give_info(boxsize_in_code_units=boxsize_in_code_units,
                   time_in_code_units=1,
                   snapnum=0)

    snap.set_volumes(vol)
    snap.set_positions(pos)
    snap.set_data(mass, '0_Masses')

    writer = pa.PaicosWriter(snap, basedir=f"{pa.data_dir}test_data", basename='paicos_file')
    for key in snap:
        writer.write_data(key, snap[key])
    writer.finalize()

    # Delete then read previously saved snapshot
    del snap
    snap = pa.GenericSnapshot(f"{pa.data_dir}test_data/paicos_file_000.hdf5")

    snap.set_volumes(vol)
    snap.set_positions(pos)

    snap.set_data(mass, '0_Masses')

# Test the projector

orientation = pa.Orientation(normal_vector=[1, 0, 0], perp_vector1=[0, 1, 0])

if True:
    widths = np.array([2000, 2000, 2000.])
    if True:
        projector = pa.Projector(snap, center, widths, orientation, npix=512,
                                 make_snap_with_selection=False)
    else:
        projector = pa.TreeProjector(snap, center, widths, orientation, npix=128, npix_depth=80,
                                     make_snap_with_selection=False)
    Masses = projector.project_variable('0_Masses')
    Volume = projector.project_variable('0_Volume')

    rho = Masses / Volume
    extent = projector.centered_extent

    image_file = pa.ArepoImage(projector, basedir=pa.data_dir + 'test_data',
                               basename='generic_sim')
    image_file.save_image('0_Masses', Masses)
    image_file.save_image('0_Volume', Volume)
    image_file.finalize()

    # Make a histogram
    snap['0_Density'] = snap['0_Masses'] / snap['0_Volume']
    rho_vol = pa.Histogram2D(snap, snap['0_Density'], snap['0_Volume'],
                             bins_x=50,
                             bins_y=50, logscale=True)
    plt.figure(2)
    plt.clf()

    if pa.settings.use_units:
        plt.pcolormesh(rho_vol.centers_x.value, rho_vol.centers_y.value,
                       rho_vol.hist2d.value, norm=LogNorm())
        plt.xlabel(rho_vol.centers_x.label('\\rho'))
        plt.ylabel(rho_vol.centers_y.label('V'))
    else:
        plt.pcolormesh(rho_vol.centers_x, rho_vol.centers_y, rho_vol.hist,
                       norm=LogNorm())
    plt.title('paicos Histogram2D')
    if rho_vol.logscale:
        plt.xscale('log')
        plt.yscale('log')

    image = pa.ImageReader(f"{pa.data_dir}test_data/generic_sim_000.hdf5")
else:
    widths = np.array([2000, 2000, 0.0])
    slicer = pa.Slicer(snap, center, widths, orientation, npix=128)
    rho = slicer.slice_variable(snap['0_Masses'] / snap['0_Volume'])
    extent = slicer.centered_extent

if pa.settings.use_units:
    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1)
    im = axes.imshow(rho.value, origin='lower',
                     extent=extent.value, norm=LogNorm())
    axes.set_xlabel(extent.label())
    axes.set_ylabel(extent.label())
    # Add a colorbar
    cbar = plt.colorbar(im, fraction=0.025, pad=0.04)

    # Set the labels. The units for the labels are here set using the .label method
    # of the PaicosQuantity. This internally uses astropy functionality and is
    # mainly a convenience function.
    cbar.set_label(rho.label('\\rho'))
else:
    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1)
    im = axes.imshow(rho, origin='lower',
                     extent=extent, norm=LogNorm())
    # Add a colorbar
    cbar = plt.colorbar(im, fraction=0.025, pad=0.04)
plt.show()
