"""Utility plotting functions in matplotlib."""
import numpy as np


def plot_yx(ax, **kwargs):
    """Plot the y=x line within a matplotlib axis."""
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against each other
    ax.plot(lims, lims, **kwargs)


def debox(ax):
    """Remove the top and right spines of a plot."""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def label_multipanel(axs, labels, xoff=-0.05, yoff=1.14, **kwargs):
    """Labeling multiple axes with text labels."""
    assert len(axs) == len(labels)
    for i, lbl in enumerate(labels):
        axs[i].text(xoff, yoff, lbl, transform=axs[i].transAxes, **kwargs)
