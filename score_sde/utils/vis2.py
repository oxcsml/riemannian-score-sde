import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # s
import matplotlib.colors as mcolors
import logging
import os
import jax.numpy as jnp

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

log = logging.getLogger()
verbose = False
save_grid = False


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
    theta = jnp.linspace(eps, jnp.pi - eps, N // 2)
    phi = jnp.linspace(eps, 2 * jnp.pi - eps, N)

    Theta, Phi = jnp.meshgrid(theta, phi)
    Theta = Theta.reshape(-1, 1)
    Phi = Phi.reshape(-1, 1)
    xs = jnp.concatenate([
        jnp.sin(Theta) * jnp.cos(Phi),
        jnp.sin(Theta) * jnp.sin(Phi), 
        jnp.cos(Theta)
    ], axis=-1)
    return xs, Theta, Phi, theta, phi


def earth_plot(log_prob, dataset, N, azimuth=170):
    """ generate earth plots with model density or integral paths aka streamplot  """
    cmap = sns.cubehelix_palette(light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True)

    # parameters
    # azimuth_dict = {"QuakesBig": 70, "Fire": 50, "Flood": 60, "Volerup": 170}
    # azimuth = azimuth_dict[str(args.dataset)] if azimuth is None else azimuth
    polar = 30
    azimuth, polar = 170, 0
    projs = ["ortho"]

    # if issubclass(type(dataset), Empirical):
    samples = dataset.dataset.dataset.data#.to(device)
    samples = latlon_from_cartesian(samples) * 180 / math.pi

    # build spherical (lat, lon) grid and compute pdf
    # N_lon, N_lat = N, N // 2
    # lon = jnp.linspace(-180, 180, N_lon)#.to(device)
    # lat = jnp.linspace(-90 + eps, 90 - eps, N_lat)#.to(device)
    # latlat, lonlon = jnp.meshgrid(lat, lon)
    # xs_latlon = jnp.concatenate([latlat.reshape((-1, 1)), lonlon.reshape((-1, 1))], axis=-1)
    # xs = cartesian_from_latlon(xs_latlon / 180 * math.pi)
    xs, _, _, theta, phi = get_spherical_grid(N, eps=0.)
    lat = 180 / jnp.pi * (theta - jnp.pi / 2)
    lon = 180 / jnp.pi * (phi - jnp.pi)
    # print("xs", xs[:5, :])
    # print(jnp.linalg.norm(xs[:5, :], axis=-1))

    fs = log_prob(xs).reshape((lat.shape[0], lon.shape[0]))

    # create figure with earth features
    dpi = 150
    # fig = plt.figure(figsize=(5, 5), dpi=dpi)
    fig = plt.figure(figsize=(700 / dpi, 700 / dpi), dpi=dpi)
    for i, proj in enumerate(projs):
        if proj == "ortho":
            projection = ccrs.Orthographic(azimuth, polar)
        elif proj == "robinson":
            projection = ccrs.Robinson(central_longitude=0)
        else:
            raise Exception("Invalid proj {}".format(proj))
        ax = fig.add_subplot(1, 1, 1, projection=projection, frameon=True)
        ax.set_global()

       
        # earth features
        # ax.add_feature(cfeature.LAND, zorder=0, facecolor="#595959")
        ax.add_feature(cfeature.LAND, zorder=0, facecolor="#e0e0e0")

        extent = (-180, 180, -90, 90)
        # ax.contourf(lon, lat, fs, levels=900, alpha=0.8, vmin=0.0, vmax=2.0, transform=ccrs.PlateCarree(), antialiased=True, cmap=cmap)
        norm = mcolors.PowerNorm(0.5)
        # norm = mcolors.Normalize(vmin=np.min(fs), vmax=np.max(fs))
        # print(fs.min().item(), np.median(fs).item(), fs.mean().item(), fs.max().item())
        fs = np.array(fs)
        # fs = norm(fs)
       
        vmin, vmax = 0.2, 1.0
        ax.contourf(lon, lat, fs, transform=ccrs.PlateCarree(), alpha=0.8, antialiased=True, cmap=cmap)
        levels = np.linspace(vmin, vmax, 900)
        colors = cmap(np.linspace(vmin, vmax, 900))
        colors[:, -1] = 0.0
        # cs = ax.contourf(
        #     lon,
        #     lat,
        #     fs,
        #     levels=levels,
        #     alpha=0.8,
        #     transform=ccrs.PlateCarree(),
        #     antialiased=True,
        #     vmin=vmin,
        #     vmax=vmax,
        #     cmap=cmap,
        #     # colors=colors
        #     extend="both",
        # )
        # cs.cmap.set_over(colors[-1, :])
        # cs.cmap.set_under("blue")
        # for col in ax.collections[0:1]:
        #     col.set_alpha(0)
        # ax.collections[10].set_alpha(0)

        # add scatter plots of the dataset
        # if issubclass(type(target), Empirical):
        colors = sns.color_palette("hls", 2)
        samples = np.array(samples)
        points = projection.transform_points(ccrs.Geodetic(), samples[:, 1], samples[:, 0])
        ax.scatter(points[:, 0], points[:, 1], s=1.0 / 2, c=[colors[1]], alpha=0.5)

        # plt.close(fig)
    return fig
