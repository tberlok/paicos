import paicos as pa
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from paicos import root_dir

pa.use_units(True)

snap = pa.Snapshot(root_dir + '/data', 247)
center = snap.Cat.Group['GroupPos'][0]

T = snap['0_Temperatures']
if pa.units.enabled:
    rho = snap['0_Density'].to_physical.astro
    M = snap['0_Masses'].to_physical.astro
    V = snap['0_Volumes'].to_physical.astro
else:
    rho = snap['0_Density']
    M = snap['0_Masses']
    V = snap['0_Volumes']

# Set up bins
bins_T = [T.min(), T.max()/10, 200]
bins_rho = [rho.min(), rho.max()*1e-4, 300]

# Create histogram object
rhoT = pa.Histogram2D(bins_rho, bins_T, logscale=True)

# Make 2D histogram
hist = rhoT.make_histogram(rho, T, weights=M, normalize=True)

plt.figure(1)
plt.clf()

if pa.units.enabled:
    plt.pcolormesh(rhoT.centers_x.value, rhoT.centers_y.value,
                   rhoT.hist, norm=LogNorm())
    plt.xlabel(rhoT.centers_x.label('\\rho'))
    plt.ylabel(rhoT.centers_y.label('T'))
else:
    plt.pcolormesh(rhoT.centers_x, rhoT.centers_y, rhoT.hist,
                   norm=LogNorm())
plt.title('paicos Histogram2D')
if rhoT.logscale:
    plt.xscale('log')
    plt.yscale('log')
cbar = plt.colorbar()
cbar.set_label(rhoT.get_colorlabel(r'\rho', 'T', 'M'))
plt.title('hist2d paicos, pcolormesh')
plt.show()
