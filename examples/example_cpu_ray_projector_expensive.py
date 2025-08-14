import paicos as pa
import numpy as np
pa.use_units(False)

pa.numthreads(pa.settings.max_threads)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

snap = pa.Snapshot(pa.data_dir, 247, basename='snap', load_catalog=True)

plt.rc('image', origin='lower', cmap='RdBu_r', interpolation='None')
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=3, sharex='col', sharey='col')

widths = [snap.box/2, snap.box/2, snap.box/2]
center = [snap.box / 2, snap.box / 2, snap.box / 2]

widths = [20000, 20000, 20000]
center = snap.Cat.Group['GroupPos'][0]

r = pa.RayProjector(snap, center, widths, 'z', npix=128, tol=0.05, verbose=False, timing=True, do_pre_selection=True)

density = r.project_variable('0_Density', additive=False, timing=True)

t = pa.TreeProjector(snap, center, widths, 'z', npix=128, tol=0.05, verbose=False, timing=True)

t_density = t.project_variable('0_Density', additive=False, timing=True)

# Make a plot
axes[0].imshow(t_density, origin='lower', extent=t.extent, norm=LogNorm())
axes[1].imshow(density, origin='lower', extent=r.extent, norm=LogNorm())
axes[2].imshow((density - t_density) / density, origin='lower', extent=r.extent)

axes[0].set_title('TreeProjector')
axes[1].set_title('RayProjector')
axes[2].set_title('Rel. difference')
plt.show()
