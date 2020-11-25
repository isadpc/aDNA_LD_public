"""Utility plotting functions in matplotlib."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

    
def colorbar(mappable, **kwargs):
    """Creating a colorbar with more gusto."""
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, **kwargs)
    plt.sca(last_axes)
    return cbar

def label_multipanel(axs, labels, xoff=-0.05, yoff=1.14, **kwargs):
    """Labeling multiple axes with text labels."""
    assert len(axs) == len(labels)
    for i, lbl in enumerate(labels):
        axs[i].text(xoff, yoff, lbl, transform=axs[i].transAxes, **kwargs)
