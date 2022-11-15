import numpy as np


class Histogram:
    """
    This is a brief class for efficient computation of many histograms
    using the same x-variables and binning but different weights.
    The heavy part of the computation is stored in idigit
    """
    def __init__(self, x_variable, bins):
        self.idigit = np.digitize(x_variable, bins)
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
    from paicos import ArepoImage

    snap = arepo_snap.snapshot('../data', 247)
    center = snap.Cat.Group['GroupPos'][0]

    snap.load_data(0, 'Coordinates')
    pos = snap.P['0_Coordinates']

    r = np.sqrt(np.sum((pos-center[None, :])**2., axis=1))

    r_max = 11000
    index = r < r_max

    bins = np.linspace(0, r_max, 1000)
    # bins = np.logspace(-2, np.log10(r_max), 1000)

    h_r = Histogram(r[index], bins)

    for key in ['Masses', 'MagneticField', 'MagneticFieldDivergence']:
        snap.load_data(0, key)

    snap.get_volumes()
    snap.get_temperatures()

    B2 = np.sum((snap.P['0_MagneticField'])**2, axis=1)
    Volumes = snap.P['0_Volumes']
    Masses = snap.P['0_Masses']

    B2TimesVolume = h_r.hist((B2*Volumes)[index])
    Volumes = h_r.hist(Volumes[index])
    TTimesMasses = h_r.hist((Masses*snap.P['0_Temperatures'])[index])
    Masses = h_r.hist(Masses[index])

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=3, sharex=True)
    axes[0].loglog(h_r.bin_centers, Masses/Volumes)
    axes[1].loglog(h_r.bin_centers, B2TimesVolume/Volumes)
    axes[2].loglog(h_r.bin_centers, TTimesMasses/Masses)

    plt.show()
