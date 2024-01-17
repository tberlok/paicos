import paicos as pa
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

snap = pa.Snapshot(pa.root_dir + '/data', 247)
center = snap.Cat.Group['GroupPos'][0]

width_vec = (
    [0.0, 20000, 20000],
    [20000, 0.0, 20000],
    [20000, 20000, 0.0],
)

plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=3, nrows=3)
for ii, direction in enumerate(['x', 'y', 'z']):
    if direction == 'x':
        orientation = pa.Orientation(normal_vector=[1, 0, 0], perp_vector1=[0, 1, 0])
    elif direction == 'y':
        orientation = pa.Orientation(normal_vector=[0, 1, 0], perp_vector1=[0, 0, 1])
    elif direction == 'z':
        orientation = pa.Orientation(normal_vector=[0, 0, 1], perp_vector1=[1, 0, 0])

    widths = [20000, 20000, 0.0]
    slicer = pa.Slicer(snap, center, widths, orientation, npix=512)
    slicer_dir = pa.Slicer(snap, center, width_vec[ii], direction, npix=512)

    image_file = pa.ArepoImage(slicer, basedir=pa.root_dir + 'test_data',
                               basename=f'slice_{direction}')

    Density = slicer.slice_variable(snap['0_Density'])
    Temperatures = slicer.slice_variable('0_Temperatures')

    image_file.save_image('0_Density', Density)

    # Move from temporary filename to final filename
    image_file.finalize()

    # Create a new image object in amend mode
    image_file = pa.ArepoImage(slicer, basedir=pa.root_dir + 'test_data',
                               basename=f'slice_{direction}',
                               mode='a')

    # Now add the temperatures as well
    image_file.save_image('0_Temperatures', Temperatures)

    # Make a plot
    extent = slicer.centered_extent
    # if pa.settings.use_units:
    #     axes[ii].imshow(Density.value, origin='lower',
    #                     extent=extent.value, norm=LogNorm())
    # else:
    #     axes[ii].imshow(Density, origin='lower',
    #                     extent=extent, norm=LogNorm())

    Density_dir = slicer_dir.slice_variable('0_Density')
    axes[0, ii].imshow(Density_dir.value,
                       origin='lower', norm=LogNorm())
    axes[1, ii].imshow(Density.value, origin='lower', norm=LogNorm())
    axes[2, ii].imshow(np.abs(Density_dir.value - Density.value),
                       origin='lower')#, norm=LogNorm())

# Example of how to read the image files
im = pa.ImageReader(pa.root_dir + 'test_data', 247, 'slice_x')

plt.show()
