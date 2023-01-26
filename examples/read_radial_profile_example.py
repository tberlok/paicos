import paicos as pa
import matplotlib.pyplot as plt
import numpy as np

# Simply read the radial file using the standard reader
pro_simple = pa.PaicosReader(pa.root_dir + 'data', 247,
                             basename='radial', load_all=True)


# Write a custom reader for our radial profiles

class RadialReader(pa.PaicosReader):
    """
    A quick custom reader for radial profiles.
    This reader gets the densities and the weighted variables of interest
    """
    def __init__(self, basedir, snapnum, basename="radial", load_all=True):

        # The PaicosReader class takes care of most of the loading
        super().__init__(basedir, snapnum, basename=basename,
                         load_all=load_all)

        # Get the interesting profiles
        keys = list(self.keys())

        keys = []
        for key in list(self.keys()):
            if not isinstance(self[key], dict):
                keys.append(key)

        for key in keys:
            if 'Times' in key:
                # Keys of the form 'MagneticFieldSquaredTimesVolume'
                # are split up
                start, end = key.split('Times')
                if (end in keys):
                    self[start] = self[key]/self[end]
                    del self[key]
                elif (start[0:2] + end in keys):
                    self[start] = self[key]/self[start[0:2] + end]
                    del self[key]

        # Calculate density if we have both masses and volumes
        for p in ['', '0_']:
            if (p + 'Masses' in keys) and (p + 'Volume' in keys):
                self[p + 'Density'] = self[p+'Masses']/self[p+'Volume']

        # For dark matter we use the bin volumes
        for p in [str(i) + '_' for i in range(1, 5)]:
            self[p + 'Density'] = self[p+'Masses']/self['bin_volumes']


pro = RadialReader(pa.root_dir + 'data', 247)

# Make a density plot of the varius particle types
plt.figure(1)
plt.clf()
for p in range(5):
    pstr = str(p)
    plt.loglog(pro['bin_centers'].astro, pro[pstr + '_Density'].cgs, label=pstr)

plt.xlabel(pro['bin_centers'].astro.label(r'\mathrm{radius}'))
plt.ylabel(pro['0_Density'].cgs.label(r'\rho'))
plt.legend(frameon=False)


# Make a plot of gas density, temperature and magnetic field
plt.figure(2)
plt.clf()
fig, axes = plt.subplots(num=2, ncols=3, sharex=True)
centers = pro['bin_centers'].astro.to_physical
rho = pro['0_Density'].cgs.to_physical
axes[0].loglog(centers, rho)
axes[0].set_ylabel(rho.label("\\rho"))

axes[1].loglog(centers, pro['0_Temperatures'])
axes[1].set_ylabel(pro['0_Temperatures'].label("T"))

B = np.sqrt(pro['0_MagneticFieldSquared']).to('uG').to_physical
axes[2].loglog(centers, B)
axes[2].set_ylabel(B.label("B"))

for ii in range(3):
    axes[ii].set_xlabel(centers.label(r'\mathrm{radius}'))


# Finally, let us make plot of the 10 most massive halos
plt.figure(3)
plt.clf()
R = pro['Group']['Group_R_Crit200'].astro
M = pro['Group']['Group_M_Crit200'].astro
plt.semilogy(R, M, '.')
plt.xlabel(R.label(r'R_{200\mathrm{c}}'))
plt.ylabel(M.label(r'M_{200\mathrm{c}}'))

plt.show()
