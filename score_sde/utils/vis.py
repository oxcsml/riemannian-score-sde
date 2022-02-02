import math
import importlib

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere

import jax
from jax import numpy as jnp
import numpy as np

try:
    plt.switch_backend("MACOSX")
except ImportError as error:
    plt.switch_backend("agg")
import seaborn as sns

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as error:
    pass


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

    return ax


def sphere_plot(ax, color="grey"):
    # assert manifold.dim == 2
    radius = 1.
    # set_aspect_equal_3d(ax)
    n = 200
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, linewidth=0, alpha=0.2)

    return ax

    # ax.set_xlim3d(-radius, radius)
    # ax.set_ylim3d(-radius, radius)
    # ax.set_zlim3d(-radius, radius)


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


def latlon_from_cartesian(points):
    r = jnp.linalg.norm(points, axis=-1)
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    lat = -jnp.arcsin(z / r)
    lon = jnp.arctan2(y, x)
    lon = jnp.where(lon > 0, lon - math.pi, lon + math.pi)
    return jnp.concatenate([jnp.expand_dims(lat, -1), jnp.expand_dims(lon, -1)], axis=-1)


def get_spherical_grid(N, eps=0.):
    lat = jnp.linspace(-90 + eps, 90 - eps, N // 2)
    lon = jnp.linspace(-180 + eps, 180 - eps, N)
    Lat, Lon = jnp.meshgrid(lat, lon)
    latlon_xs = jnp.concatenate([Lat.reshape(-1, 1), Lon.reshape(-1, 1)], axis=-1)
    spherical_xs = (
            jnp.pi * (latlon_xs / 180.0) + jnp.array([jnp.pi / 2, jnp.pi])[None, :]
        )
    xs = Hypersphere(2).spherical_to_extrinsic(spherical_xs)
    return xs, lat, lon


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


def earth_plot(cfg, log_prob, train_ds, test_ds, N, azimuth=None):
    """ generate earth plots with model density or integral paths aka streamplot  """
    has_cartopy = importlib.find_loader('cartopy')
    if not has_cartopy:
        return

    # parameters
    azimuth_dict = {"earthquake": 70, "fire": 50, "floow": 60, "volecanoe": 170}
    azimuth = azimuth_dict[str(cfg.dataset.name)] if azimuth is None else azimuth
    polar = 30
    projs = ["ortho", "robinson"]
    # projs = ["ortho"]

    xs, lat, lon = get_spherical_grid(N, eps=0.)
    # ts = [0.01, 0.05, cfg.sde.tf]
    ts = [cfg.sde.tf]
    figs = []
  
    for t in ts:
        print(t)
        fs = log_prob(xs, t).reshape((lat.shape[0], lon.shape[0]), order="F")
        fs = jnp.exp(fs)
        # norm = mcolors.PowerNorm(3.)  # NOTE: tweak that value
        norm = mcolors.PowerNorm(.2)  # N=500
        fs = np.array(fs)
        # print(np.min(fs).item(), jnp.quantile(fs, np.array([0.1, 0.5, 0.9])), np.max(fs).item())
        fs = norm(fs)
        # print(np.min(fs).item(), jnp.quantile(fs, np.array([0.1, 0.5, 0.9])), np.max(fs).item())

        # create figure with earth features
        for i, proj in enumerate(projs):
            print(proj)
            fig = plt.figure(figsize=(5, 5), dpi=300)
            if proj == "ortho":
                projection = ccrs.Orthographic(azimuth, polar)
            elif proj == "robinson":
                projection = ccrs.Robinson(central_longitude=0)
            else:
                raise Exception("Invalid proj {}".format(proj))
            ax = fig.add_subplot(1, 1, 1, projection=projection, frameon=True)
            ax.set_global()

            # earth features
            ax.add_feature(cfeature.LAND, zorder=0, facecolor="#e0e0e0")

            vmin, vmax = 0., 1.0
            # n_levels = 900
            n_levels = 200
            levels = np.linspace(vmin, vmax, n_levels)
            cmap = sns.cubehelix_palette(light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True)
            # cmap = sns.cubehelix_palette(as_cmap=True)
            cs = ax.contourf(
                lon,
                lat,
                fs,
                levels=levels,
                # alpha=0.8,
                transform=ccrs.PlateCarree(),
                antialiased=True,
                # vmin=vmin,
                # vmax=vmax,
                cmap=cmap,
                extend="both",
            )
        
            alpha_gradient = np.linspace(0, 1, len(ax.collections))
            for col, alpha in zip(ax.collections, alpha_gradient):
                col.set_alpha(alpha)
            # for col in ax.collections[0:1]:
                # col.set_alpha(0)

            # add scatter plots of the dataset
            colors = sns.color_palette("hls", 8)
            # colors = sns.color_palette()
            train_idx = train_ds.dataset.indices
            test_idx = test_ds.dataset.indices
            samples = train_ds.dataset.dataset.data
            samples = np.array(latlon_from_cartesian(samples)) * 180 / math.pi
            points = projection.transform_points(ccrs.Geodetic(), samples[:, 1], samples[:, 0])
            ax.scatter(points[train_idx, 0], points[train_idx, 1], s=0.2, c=[colors[5]], alpha=0.2)
            ax.scatter(points[test_idx, 0], points[test_idx, 1], s=0.2, c=[colors[0]], alpha=0.2)
            # plt.close(fig)
            figs.append(fig)

    return figs
