import numpy as np


class Histogram:
    """
    This is a brief class for efficient computation of many histograms
    using the same x-variables and binning but different weights.
    The heavy part of the computation is stored in idigit
    """
    def __init__(self, x_variable, bins, verbose=False):
        self.verbose = verbose

        if self.verbose:
            import time
            t = time.time()
            print('Digitize for histogram begun')

        self.idigit = np.digitize(x_variable, bins)
        if self.verbose:
            print('This took {:1.2f} seconds'.format(time.time()-t))
        self.bins = bins
        self.bin_centers = 0.5*(bins[1:] + bins[:-1])

    def hist(self, weights):
        from paicos import get_hist_from_weights_and_idigit
        f = get_hist_from_weights_and_idigit(self.bins.shape[0],
                                             weights.astype(np.float64),
                                             self.idigit)
        return f


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from paicos import arepo_snap

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=3, sharex=True)
    for ii in range(2):
        snap = arepo_snap.snapshot('../data', 247)
        center = snap.Cat.Group['GroupPos'][0]

        snap.load_data(0, 'Coordinates')
        pos = snap.P['0_Coordinates']

        r = np.sqrt(np.sum((pos-center[None, :])**2., axis=1))

        r_max = 10000
        index = r < r_max*1.1

        bins = np.linspace(0, r_max, 150)
        # bins = np.logspace(-2, np.log10(r_max), 1000)

        for key in ['Masses', 'MagneticField', 'MagneticFieldDivergence']:
            snap.load_data(0, key)

        snap.get_volumes()
        snap.get_temperatures()

        B2 = np.sum((snap.P['0_MagneticField'])**2, axis=1)
        Volumes = snap.P['0_Volumes']
        Masses = snap.P['0_Masses']

        if ii == 0:
            h_r = Histogram(r[index], bins, verbose=True)
            B2TimesVolume = h_r.hist((B2*Volumes)[index])
            Volumes = h_r.hist(Volumes[index])
            TTimesMasses = h_r.hist((Masses*snap.P['0_Temperatures'])[index])
            Masses = h_r.hist(Masses[index])

            axes[0].loglog(h_r.bin_centers, Masses/Volumes)
            axes[1].loglog(h_r.bin_centers, B2TimesVolume/Volumes)
            axes[2].loglog(h_r.bin_centers, TTimesMasses/Masses)
        else:
            B2TimesVolume, edges = np.histogram(r[index], weights=(B2*Volumes)[index], bins=bins)
            Volumes, edges = np.histogram(r[index], weights=Volumes[index], bins=bins)
            TTimesMasses, edges = np.histogram(r[index], weights=(Masses*snap.P['0_Temperatures'])[index], bins=bins)
            Masses, edges = np.histogram(r[index], weights=Masses[index], bins=bins)
            bin_centers = 0.5*(edges[1:] + edges[:-1])

            axes[0].loglog(bin_centers, Masses/Volumes, '--')
            axes[1].loglog(bin_centers, B2TimesVolume/Volumes, '--')
            axes[2].loglog(bin_centers, TTimesMasses/Masses, '--')

    plt.show()
