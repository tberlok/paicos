import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from paicos import Snapshot
from paicos import Projector, NestedProjector
import numpy as np
from paicos import root_dir

snap = Snapshot(root_dir + '/data', 247)
center = snap.Cat.Group['GroupPos'][0]
R200c = snap.Cat.Group['Group_R_Crit200'][0]

snap.get_volumes()
snap.load_data(0, "Coordinates")
snap.load_data(0, "Masses")
snap.load_data(0, "Density")

snap.P['0_Coordinates'] = snap.P['0_Coordinates'][:9]
snap.P['0_Volumes'] = snap.P['0_Volumes'][:9]
snap.P['0_Masses'] = snap.P['0_Masses'][:9]
snap.P['0_Density'] = snap.P['0_Density'][:9]

for jj in range(3):
    for kk in range(3):
        ii = jj*3 + kk
        snap.P['0_Volumes'][ii] = 4*np.pi/3*(snap.box/2/(10*(ii+1)))**3
        snap.P['0_Coordinates'][ii][0] = snap.box*1/4
        snap.P['0_Coordinates'][ii][1] = snap.box*1/4
        snap.P['0_Coordinates'][ii][2] = snap.box*1/2
        snap.P['0_Masses'][ii] = snap.P['0_Volumes'][ii]
        snap.P['0_Density'][ii] = 1.0
        snap.P['0_Coordinates'][ii][0] += jj*snap.box*1/4
        snap.P['0_Coordinates'][ii][1] += kk*snap.box*1/4


plt.rc('image', origin='lower', cmap='RdBu_r', interpolation='None')
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=3,
                         sharex='col', sharey='col')#, sharey=True)

widths = [snap.box, snap.box, snap.box]
center = [snap.box/2, snap.box/2, snap.box/2]
for ii, direction in enumerate(['z']):
    p_nested = NestedProjector(snap, center, widths, direction, npix=512,
                               npix_min=8)

    Masses_nested = p_nested.project_variable('Masses')
    Volume_nested = p_nested.project_variable('Volumes')

    nested_image = np.zeros_like(Masses_nested)
    nested_image[Masses_nested > 0] = Masses_nested[Masses_nested > 0]/Volume_nested[Masses_nested > 0]

    p = Projector(snap, center, widths, direction, npix=512)

    Masses = p.project_variable('Masses')
    Volume = p.project_variable('Volumes')

    normal_image = np.zeros_like(Masses)
    normal_image[Masses > 0] = Masses[Masses > 0]/Volume[Masses > 0]

    # Make a plot
    axes[0].imshow(normal_image, origin='lower',
                       extent=p.extent)
    axes[1].imshow(nested_image, origin='lower',
                       extent=p_nested.extent)
    axes[2].imshow(normal_image-nested_image, origin='lower',
                       extent=p_nested.extent)
    # print(np.max(np.abs(normal_image-nested_image)/normal_image))
    # print(np.sum(normal_image.flatten()), np.sum(nested_image.flatten()))
axes[0].set_title('Standard projection')
axes[1].set_title('nested grids')
axes[2].set_title('difference')
plt.show()
