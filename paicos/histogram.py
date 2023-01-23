import numpy as np
from . import units as pu


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
        from .cython.histogram import get_hist_from_weights_and_idigit

        if hasattr(weights, 'unit'):
            if isinstance(weights, pu.PaicosQuantity):
                factor = weights.unit_quantity
            else:
                factor = 1.0*weights.unit
            weights = weights.value
        else:
            factor = 1.0

        # compute histogram using pre-digitized x-coordinates and given weights
        f = get_hist_from_weights_and_idigit(self.bins.shape[0], weights,
                                             self.idigit)
        return f * factor
