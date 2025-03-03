"""
This module defines a class for 1D histograms.
"""
import time
import numpy as np
from astropy import units as u
from .. import util
from .. import settings
from ..cython.histogram import get_hist_from_weights_and_idigit as func


def _make_bins(bins, logscale):
    """
    Private method to calculate the edges and centers of the bins
    given lower, upper and number of bins.

    Parameters:
        bins (tuple): Tuple of lower edge, upper edge and number of
                      bins for the axis
    Returns:
        edges (array): Edges of the bins
        centers (array): Centers of the bins
    """

    lower, upper, nbins = bins
    edges, centers = _make_bins_helper(lower, upper, nbins, logscale)
    if settings.use_units:
        assert lower.unit == upper.unit
        edges = np.array(edges) * lower.unit_quantity
        centers = np.array(centers) * lower.unit_quantity
    return edges, centers


@util.remove_astro_units
def _make_bins_helper(lower, upper, nbins, logscale):
    """
    Private method to calculate the edges and centers of the bins
    in logscale or linear scale based on the class variable logscale
    Parameters:
        lower (float): Lower edge of the bin
        upper (float): Upper edge of the bin
        nbins (int): Number of bins for the axis
    Returns:
        edges (array): Edges of the bins
        centers (array): Centers of the bins
    """

    if logscale:
        lower = np.log10(lower)
        upper = np.log10(upper)

    edges = lower + np.arange(nbins + 1) * (upper - lower) / nbins
    centers = 0.5 * (edges[1:] + edges[:-1])

    if logscale:
        edges = 10**edges
        centers = 10**centers

    return edges, centers


class Histogram:
    """
    This is a brief class for efficient computation of many 1D histograms
    using the same x-variables and binning but different weights.

    The heavy part of the computation is stored in idigit
    """

    def __init__(self, x, bins, logscale=False, verbose=False):
        """
        Initialize the class.

        Parameters:
            x_variable (array): x-coordinates of the data.

            bins (array): Bin edges for the histogram.

            logscale (bool): Indicates whether to use logscale for the
                             histogram, default is True.

            verbose (bool, optional): If True, prints the time taken to
                                      digitize the x-coordinates.
        """

        if isinstance(bins, int):
            bins = [x.min(), x.max(), bins]

        self.logscale = logscale
        self.verbose = verbose

        self.edges, self.centers = _make_bins(bins, self.logscale)
        self.bin_centers = self.centers
        self.bin_edges = self.edges

        if self.verbose:
            t = time.time()
            print('Digitize for histogram begun')

        self.idigit = np.digitize(x, self.edges)
        if self.verbose:
            print(f'This took {time.time() - t:1.2f} seconds')

    def hist(self, weights=None, normalize=False):
        """
        Compute the histogram of the data.

        Parameters:
            weights (array): Weights of the data.

        Returns:
            array: Histogram of the data.
        """

        if weights is None and settings.use_units:
            weights = np.ones(self.idigit.shape[0]) * u.Unit('')
        elif weights is None and not settings.use_units:
            weights = np.ones(self.idigit.shape[0])

        if settings.use_units:
            uq = self.bin_edges.unit_quantity
            uq = uq / uq  # unitless_paicos_quanity
            hist_unit = uq * weights.unit

        get_hist_from_weights_and_idigit = util.remove_astro_units(func)

        # compute histogram using pre-digitized x-coordinates and given weights
        hist = get_hist_from_weights_and_idigit(self.edges.shape[0], weights,
                                                self.idigit)

        if settings.use_units:
            hist = hist * hist_unit

        if settings.use_units:
            edges = self.bin_edges.value
        else:
            edges = self.bin_edges

        if self.logscale:
            self.area_per_bin = np.diff(np.log10(edges))
            if settings.use_units:
                self.area_per_bin = self.area_per_bin * u.Unit('dex')
        else:
            self.area_per_bin = np.diff(edges)
            if settings.use_units:
                self.area_per_bin = self.area_per_bin * self.bin_edges.unit

        if normalize:
            total_weight = np.sum(hist)
            hist = hist / (total_weight * self.area_per_bin)

        return hist
