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


def plot_and_save(
    x0, xt, prob, grad, x0prob=None, size=20, dpi=300, out="out.jpg", color="red"
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
    # sphere.plot_heatmap(ax, pdf, n_points=16000, alpha=0.2, cmap=cmap)
    if x0 is not None:
        cax = ax.scatter(x0[:,0], x0[:,1], x0[:,2], s=50, color='green')
    if xt is not None:
        x, y, z = xt[:,0], xt[:,1], xt[:,2]
        c = prob if prob is not None else np.ones([*xt.shape[:-1]])
        cax = ax.scatter(x, y, z, s=50, c=c, cmap=cmap)
    if grad is not None:
        u, v, w = grad[:, 0], grad[:, 1], grad[:, 2]
        quiver = ax.quiver(x, y, z, u, v, w, length=.2, lw=2, normalize=False, cmap=cmap)
        quiver.set_array(c)

    plt.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)

