import paicos as pa
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

pa.use_units(True)

snap = pa.Snapshot(pa.root_dir + '/data', 247)
center = snap.Cat.Group['GroupPos'][0]

T = snap['0_Temperatures']
if pa.settings.use_units:
    rho = snap['0_Density'].to_physical.astro
    M = snap['0_Masses'].to_physical.astro
    V = snap['0_Volume'].to_physical.astro
else:
    rho = snap['0_Density']
    M = snap['0_Masses']
    V = snap['0_Volume']

# Set up bins
bins_T = [T.min(), T.max()/10, 200]
bins_rho = [rho.min(), rho.max()*1e-4, 300]

# Create histogram object
rhoT = pa.Histogram2D(snap, rho, T, weights=M, bins_x=bins_rho,
                      bins_y=bins_T, logscale=True)

# Create colorlabel
colorlabel = rhoT.get_colorlabel(r'\rho', 'T', 'M')

# Save the 2D histogram
rhoT.save(basedir=pa.root_dir + '/data', basename='rhoT_hist')

del rhoT

rhoT = pa.Histogram2DReader(pa.root_dir + '/data', 247, basename='rhoT_hist')

plt.figure(1)
plt.clf()

if pa.settings.use_units:
    plt.pcolormesh(rhoT.centers_x.value, rhoT.centers_y.value,
                   rhoT.hist2d.value, norm=LogNorm())
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
cbar.set_label(rhoT.colorlabel)
plt.title('hist2d paicos, pcolormesh')
plt.show()
