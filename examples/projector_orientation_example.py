import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from paicos import root_dir
import paicos as pa
import numpy as np

pa.use_units(True)

snap = pa.Snapshot(root_dir + '/data', 247)
center = snap.Cat.Group['GroupPos'][0]
R200c = snap.Cat.Group['Group_R_Crit200'][0].value

widths = [10000, 10000, 10000]
width_vec = (
    [2 * R200c, 20000, 10000],
    [20000, 2 * R200c, 10000],
    [20000, 10000, 2 * R200c],
)

plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=3, nrows=3)
for ii, direction in enumerate(['x', 'y', 'z']):
    if direction == 'x':
        orientation = pa.Orientation(normal_vector=[1, 0, 0], perp_vector1=[0, 1, 0])
    elif direction == 'y':
        # Somewhat weird orientation of the standard y-image, perhaps change that?
        orientation = pa.Orientation(normal_vector=[0, -1, 0], perp_vector1=[1, 0, 0])
    elif direction == 'z':
        orientation = pa.Orientation(normal_vector=[0, 0, 1], perp_vector1=[1, 0, 0])

    projector_dir = pa.Projector(snap, center, width_vec[ii], direction, npix=512)
    projector = pa.Projector(snap, center, width_vec[2], orientation, npix=512)

    projector.project_variable('0_Masses')
    density_dir = projector_dir.project_variable('0_Masses') / projector_dir.project_variable('0_Volume')
    density = projector.project_variable('0_Masses') / projector.project_variable('0_Volume')

    # Make a plot
    axes[0, ii].imshow(density_dir.value, origin='lower', norm=LogNorm())
    axes[1, ii].imshow(density.value, origin='lower', norm=LogNorm())
    axes[2, ii].imshow((density - density_dir).value, origin='lower')

plt.show()
