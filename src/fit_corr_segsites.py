"""Functions to estimate Ne and sample age from correlation in segregating sites."""

from scipy.optimize import curve_fit
from coal_cov import TwoLocusTheoryConstant


def corr_seg_sites_fit_constant(rec_rates, ta=0.0, Ne=1e4, theta=0.4):
    """Calculate fit where theta is specified (not assumed for bp window)."""
    t = ta / Ne / 2
    rhos = 4 * Ne * rec_rates
    return TwoLocusTheoryConstant._corrSASB(rhos, ta=t, theta=theta)


def corr_seg_sites_fit_constant_1kb(rec_rates, Ne, ta):
    """Fitting where mutation rate is fixed and we only look at 1kb windows."""
    L = 1e3
    mut_rate = 1.2e-8
    theta = 4.0 * Ne * mut_rate * L
    t = ta / Ne / 2.0
    rhos = 4.0 * Ne * rec_rates
    return TwoLocusTheoryConstant._corrSASB(rhos, ta=t, theta=theta)


def fit_constant_1kb(rec_rates, y, **kwargs):
    """Fitting a constant population size model with 1kb estimates."""
    popt, pcov = curve_fit(corr_seg_sites_fit_constant_1kb, rec_rates, y, **kwargs)
    return (popt, pcov)


def fit_constant(rec_rates, y, **kwargs):
    """Fit a constant population size to the data."""
    popt, pcov = curve_fit(corr_seg_sites_fit_constant, rec_rates, y, **kwargs)
    return (popt, pcov)
