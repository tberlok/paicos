import numpy as np
from paicos import units as pu


class Histogram:
    """
    This is a brief class for efficient computation of many 1D histograms
    using the same x-variables and binning but different weights.

    The heavy part of the computation is stored in idigit
    """
    def __init__(self, x_variable, bins, verbose=False):
        """
        Initialize the class.

        Parameters:
            x_variable (array): x-coordinates of the data.

            bins (array): Bin edges for the histogram.

            verbose (bool, optional): If True, prints the time taken to
                                      digitize the x-coordinates.
        """
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
        """
        Compute the histogram of the data.

        Parameters:
            weights (array): Weights of the data.

        Returns:
            array: Histogram of the data.
        """
        from paicos import get_hist_from_weights_and_idigit

        if hasattr(weights, 'unit'):
            if isinstance(weights, pu.PaicosQuantity):
                factor = weights.unit_quantity
            else:
                factor = 1.0*weights.unit
            weights = weights.value
        else:
            factor = 1.0

        # compute histogram using pre-digitized x-coordinates and given weights
        f = get_hist_from_weights_and_idigit(self.bins.shape[0],
                                             weights.astype(np.float64),
                                             self.idigit)
        return f * factor


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from paicos import Snapshot
    from paicos import root_dir
    import paicos as pa

    pa.use_units(True)

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=3, sharex=True)
    for ii in range(2):
        snap = Snapshot(root_dir + '/data', 247)
        center = snap.Cat.Group['GroupPos'][0]

        pos = snap['0_Coordinates']

        r = np.sqrt(np.sum((pos-center[None, :])**2., axis=1))

        if pa.settings.use_units:
            r_max = 10000 * r.unit_quantity
        else:
            r_max = 10000

        index = r < r_max*1.1

        bins = np.linspace(0, r_max, 150)
        # bins = np.logspace(-2, np.log10(r_max), 1000)

        B2 = np.sum((snap['0_MagneticField'])**2, axis=1)
        Volumes = snap['0_Volumes']
        Masses = snap['0_Masses']

        if ii == 0:
            h_r = Histogram(r[index], bins, verbose=True)
            B2TimesVolumes = h_r.hist((B2*Volumes)[index])
            Volumes = h_r.hist(Volumes[index])
            TTimesMasses = h_r.hist((Masses*snap['0_Temperatures'])[index])
            Masses = h_r.hist(Masses[index])

            axes[0].loglog(h_r.bin_centers, Masses/Volumes)
            axes[1].loglog(h_r.bin_centers, B2TimesVolumes/Volumes)
            axes[2].loglog(h_r.bin_centers, TTimesMasses/Masses)

            if pa.settings.use_units:
                axes[0].set_xlabel(h_r.bin_centers.label(r'\mathrm{radius}\;'))
                axes[0].set_ylabel((Masses/Volumes).label('\\rho'))
                axes[1].set_ylabel((B2TimesVolumes/Volumes).label('B^2'))
                axes[2].set_ylabel((TTimesMasses/Masses).label('T'))
        else:
            B2TimesVolumes, edges = np.histogram(r[index], weights=(B2*Volumes)[index], bins=bins)
            Volumes, edges = np.histogram(r[index], weights=Volumes[index], bins=bins)
            TTimesMasses, edges = np.histogram(r[index], weights=(Masses*snap['0_Temperatures'])[index], bins=bins)
            Masses, edges = np.histogram(r[index], weights=Masses[index], bins=bins)
            bin_centers = 0.5*(edges[1:] + edges[:-1])

            axes[0].loglog(bin_centers, Masses/Volumes, '--')
            axes[1].loglog(bin_centers, B2TimesVolumes/Volumes, '--')
            axes[2].loglog(bin_centers, TTimesMasses/Masses, '--')

    plt.show()
