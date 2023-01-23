import paicos as pa

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

snap = pa.Snapshot(pa.root_dir + '/data', 247)
center = snap.Cat.Group['GroupPos'][0]

if pa.settings.use_units:
    R200c = snap.Cat.Group['Group_R_Crit200'][0].value
else:
    R200c = snap.Cat.Group['Group_R_Crit200'][0]

width_vec = (
    [2*R200c, 10000, 10000],
    [10000, 2*R200c, 10000],
    [10000, 10000, 2*R200c],
)

plt.rc('image', origin='lower', cmap='RdBu_r', interpolation='None')
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=3, nrows=3,
                         sharex='col', sharey='col')

for ii, direction in enumerate(['x', 'y', 'z']):

    widths = width_vec[ii]
    p_nested = pa.NestedProjector(snap, center, widths, direction, npix=512)

    Masses = p_nested.project_variable('0_Masses')
    Volume = p_nested.project_variable('0_Volumes')

    nested_image = Masses/Volume

    p = pa.Projector(snap, center, widths, direction, npix=512)

    Masses = p.project_variable('0_Masses')
    Volume = p.project_variable('0_Volumes')

    normal_image = Masses/Volume

    if pa.settings.use_units:
        if ii == 0:
            vmin = normal_image.min().value
            vmax = normal_image.max().value

        # Make a plot
        axes[0, ii].imshow(normal_image.value, origin='lower',
                           extent=p.extent.value,
                           norm=LogNorm(vmin=vmin, vmax=vmax))

        axes[1, ii].imshow(nested_image.value, origin='lower',
                           extent=p_nested.extent.value,
                           norm=LogNorm(vmin=vmin, vmax=vmax))

        axes[2, ii].imshow(np.abs(normal_image-nested_image).value,
                           origin='lower',
                           extent=p_nested.extent.value,
                           norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        if ii == 0:
            vmin = normal_image.min()
            vmax = normal_image.max()

        # Make a plot
        axes[0, ii].imshow(normal_image, origin='lower',
                           extent=p.extent,
                           norm=LogNorm(vmin=vmin, vmax=vmax))

        axes[1, ii].imshow(nested_image, origin='lower',
                           extent=p_nested.extent,
                           norm=LogNorm(vmin=vmin, vmax=vmax))

        axes[2, ii].imshow(np.abs(normal_image-nested_image), origin='lower',
                           extent=p_nested.extent,
                           norm=LogNorm(vmin=vmin, vmax=vmax))

    axes[0, ii].set_title('Normal projection ({})'.format(direction))
    axes[1, ii].set_title('nested projection')
    axes[2, ii].set_title('difference')
    # print(np.max(np.abs(normal_image-nested_image)/normal_image))
    print(np.sum(normal_image.flatten()), np.sum(nested_image.flatten()))
plt.show()
