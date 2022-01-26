import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns

import geomstats.backend as gs
import geomstats.visualization as visualization

import jax
from jax import numpy as jnp
import numpy as np


def remove_background(ax):
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return ax


def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)

    plot_radius = max([abs(lim - mean_) for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean)) for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def sphere_plot(ax):
    # assert manifold.dim == 2
    radius = 1.
    set_aspect_equal_3d(ax)

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color="grey", linewidth=0, alpha=0.2)

    # ax.set_xlim3d(-radius, radius)
    # ax.set_ylim3d(-radius, radius)
    # ax.set_zlim3d(-radius, radius)


def plot(
    x0, xt, prob, grad, x0prob=None, size=10, dpi=300, out="out.jpg", color="red"
):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax = remove_background(ax)
    fig.subplots_adjust(left=-0.2, bottom=-0.2, right=1.2, top=1.2, wspace=0, hspace=0)
    # ax.view_init(elev=30, azim=45)
    ax.view_init(elev=0, azim=0)
    cmap = sns.cubehelix_palette(as_cmap=True)
    sphere = visualization.Sphere()
    sphere.draw(ax, color="red", marker=".")
    # sphere_plot(ax)
    # sphere.plot_heatmap(ax, pdf, n_points=16000, alpha=0.2, cmap=cmap)
    if x0 is not None:
        cax = ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], s=50, color="green")
    if xt is not None:
        x, y, z = xt[:, 0], xt[:, 1], xt[:, 2]
        c = prob if prob is not None else np.ones([*xt.shape[:-1]])
        cax = ax.scatter(x, y, z, s=50, vmin=0., vmax=2., c=c, cmap=cmap)
    if grad is not None:
        u, v, w = grad[:, 0], grad[:, 1], grad[:, 2]
        quiver = ax.quiver(
            x, y, z, u, v, w, length=0.2, lw=2, normalize=False, cmap=cmap
        )
        quiver.set_array(c)

    plt.colorbar(cax)
    # plt.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return fig


def setup_sphere_plot(size=10, dpi=300, elev=0, azim=45):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax = remove_background(ax)
    fig.subplots_adjust(left=-0.2, bottom=-0.2, right=1.2, top=1.2, wspace=0, hspace=0)
    # ax.view_init(elev=30, azim=45)
    ax.view_init(elev=elev, azim=azim)
    sphere = visualization.Sphere()
    sphere.draw(ax, color="red", marker=".")

    return fig, ax


def scatter_earth(x, ax=None, s=50, color="green"):
    if ax is None:
        ax = setup_sphere_plot()
    cax = ax.scatter(x[:, 0], x[:, 1], -x[:, 2], s=s, color=color)
