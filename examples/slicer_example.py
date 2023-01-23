import paicos as pa
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

snap = pa.Snapshot(pa.root_dir + '/data', 247)
center = snap.Cat.Group['GroupPos'][0]

width_vec = (
    [0.0, 10000, 10000],
    [10000, 0.0, 10000],
    [10000, 10000, 0.0],
    )

plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=3)
for ii, direction in enumerate(['x', 'y', 'z']):
    widths = width_vec[ii]
    slicer = pa.Slicer(snap, center, widths, direction, npix=512)

    image_file = pa.ArepoImage(slicer, basedir=pa.root_dir + '/data/',
                               basename='slice_{}'.format(direction))

    Density = slicer.slice_variable(snap['0_Density'])
    Temperatures = slicer.slice_variable('0_Temperatures')

    image_file.save_image('Density', Density)

    # Move from temporary filename to final filename
    image_file.finalize()

    # Make a plot
    if pa.settings.use_units:
        axes[ii].imshow(Density.value, origin='lower',
                        extent=slicer.extent.value, norm=LogNorm())
    else:
        axes[ii].imshow(Density, origin='lower',
                        extent=slicer.extent, norm=LogNorm())

plt.show()
