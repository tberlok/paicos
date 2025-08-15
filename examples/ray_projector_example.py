import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import paicos as pa
import numpy as np

pa.use_units(True)

snap = pa.Snapshot(pa.data_dir, 247)
center = snap.Cat.Group['GroupPos'][0]
R200c = snap.Cat.Group['Group_R_Crit200'][0].value
widths = [10000, 10000, 2 * R200c]

projector = pa.RayProjector(snap, center, widths, 'z', npix=512, tol=1, timing=True)
extent = projector.centered_extent.to_physical.to('Mpc')

variables = ["0_Masses", "0_Volume", "0_Density", "0_Temperatures", "0_MagneticFieldSquaredTimesVolume"]
additive = [True, True, False, False, True]
images = projector.project_variables(variables, additive, timing=True)
Masses = images[0]
Volume = images[1]
Density = images[2]
T = images[3].to('keV')
B = (np.sqrt(images[4] / images[1])).to_physical.to('uG')

# Make plots

# --- Figure 1 ---
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=2, sharey=True)
im0 = axes[0].imshow(np.array((Masses / Volume)), origin='lower', extent=extent.value, norm=LogNorm())
im1 = axes[1].imshow(np.array(Density), origin='lower', extent=extent.value, norm=LogNorm())

for ax in axes:
    ax.set_xlabel(extent.label('x'))
axes[0].set_ylabel(extent.label('y'))

# Colorbars above images
for ax, im, label in zip(axes, [im0, im1], [(Masses / Volume).label(r'M / V'), Density.label(r'\rho')]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(label)
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')


# --- Figure 2 ---
plt.figure(2)
plt.clf()
fig, axes = plt.subplots(num=2, ncols=2, sharey=True)
im0 = axes[0].imshow(np.array((B)), origin='lower', extent=extent.value, norm=LogNorm())
im1 = axes[1].imshow(np.array(T), origin='lower', extent=extent.value, norm=LogNorm())

for ax in axes:
    ax.set_xlabel(extent.label('x'))
axes[0].set_ylabel(extent.label('y'))

# Colorbars above images
for ax, im, label in zip(axes, [im0, im1], [B.label(r'B'), T.label(r'T')]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(label)
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

plt.show()
