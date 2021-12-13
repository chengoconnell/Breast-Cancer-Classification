import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from datetime import datetime


def histogram_projected_data(projected_inputs, targets, classes, threshold):
    """
    Given a vector of projected inputs, their corresponding targets, and classification threshold,
    plots them in a histogram.

    :param projected_inputs: 1-d numpy array of projected inputs
    :param targets: 1-d numpy array of target values (classes)
    :param classes: class values, e.g. 0 and 1
    :param threshold: float classification threshold

    :return None: saves the produced plot
    """
    ax = plot_class_histograms(projected_inputs, targets)
    # label x axis
    ax.set_xlabel(r"$\mathbf{w}^T\mathbf{x}$")
    ax.set_title("Projected Data: fisher")

    # add legend
    if not classes is None:
        ax.legend(classes)

    # add vertical line for threshold
    ax.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    ax.text(threshold * 1.05, max_ylim * 0.9, 'Classification \nthreshold: {:.2f}'.format(threshold))

    # save the plot to disk
    plt.savefig(os.path.join("plots", "fisher",
                             "projected_class_histogram_{}.png".format(datetime.now().strftime("%d-%m-%Y %H %M %S"))))


def plot_class_histograms(inputs, class_assignments, bins=20, colors=None, ax=None):
    """
    Plots histograms of 1d input data, split according to class

    parameters
    ----------
    inputs - 1d vector of input values (array-like)
    class_assignments - 1d vector of class values as integers (array-like)
    colors (optional) - a vector of colors one per class
    ax (optional) - pass in an existing axes object (otherwise one will be
        created)
    """
    class_ids = np.unique(class_assignments)
    num_classes = len(class_ids)

    # calculate a good division of bins for the whole data-set
    _, bins = np.histogram(inputs, bins=bins)

    # create an axes object if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # create colors for classes if needed
    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, num_classes))

    # plot histograms
    for i, class_id in enumerate(class_ids):
        class_inputs = inputs[class_assignments == class_id]
        ax.hist(class_inputs, bins=bins, color=colors[i], alpha=0.6)
    return ax


def overlay_2d_gaussian_contour(ax, mu, Sigma, num_grid_points=60, levels=10):
    """
    Overlays the contours of a 2d-gaussian with mean, mu, and covariance matrix
    Sigma onto an existing set of axes.

    parameters
    ----------
    ax -- a matplotlib.axes.Axes object on which to plot the contours
    mu -- a 2-vector mean of the distribution
    Sigma -- the (2x2)-covariance matrix of the distribution.
    num_grid_points (optional) -- the number of grid_points along each dimension
      at which to evaluate the pdf
    levels (optional) -- the number of contours (or the function values at which
      to draw contours)
    """
    # generate num_grid_points grid-points in each dimension
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xpoints = np.linspace(xmin, xmax, num_grid_points)
    ypoints = np.linspace(ymin, ymax, num_grid_points)

    # meshgrid produces two 2d arrays of the x and y coordinates
    xgrid, ygrid = np.meshgrid(xpoints, ypoints)

    # Pack xgrid and ygrid into a single 3-dimensional array
    pos = np.empty(xgrid.shape + (2,))
    pos[:, :, 0] = xgrid
    pos[:, :, 1] = ygrid

    # create a distribution over the random variable
    rv = stats.multivariate_normal(mu, Sigma)

    # evaluate the rv probability density at every point on the grid
    prob_density = rv.pdf(pos)

    # plot the contours
    ax.contour(xgrid, ygrid, prob_density, levels=levels)
