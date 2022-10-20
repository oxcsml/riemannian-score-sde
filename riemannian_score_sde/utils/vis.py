import math
import importlib
from functools import partial

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import seaborn as sns

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = ["Computer Modern Roman"]
# plt.rcParams.update({"font.size": 20})

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.hyperbolic import Hyperbolic, PoincareBall, Hyperboloid
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.special_orthogonal import (
    _SpecialOrthogonalMatrices,
    _SpecialOrthogonal3Vectors,
)
from geomstats.geometry.product_manifold import ProductSameManifold

import jax
from jax import numpy as jnp
import numpy as np
from scipy.stats import norm

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

    plot_radius = max(
        [
            abs(lim - mean_)
            for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
            for lim in lims
        ]
    )

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

    return ax


def get_sphere_coords():
    radius = 1.0
    # set_aspect_equal_3d(ax)
    n = 200
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z


def sphere_plot(ax, color="grey"):
    # assert manifold.dim == 2
    x, y, z = get_sphere_coords()
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
    # lon = jnp.where(lon > 0, lon - math.pi, lon + math.pi)
    return jnp.concatenate([jnp.expand_dims(lat, -1), jnp.expand_dims(lon, -1)], axis=-1)


def cartesian_from_latlong(points):
    lat = points[..., 0]
    lon = points[..., 1]

    x = jnp.cos(lat) * jnp.cos(lon)
    y = jnp.cos(lat) * jnp.sin(lon)
    z = jnp.sin(lat)

    return jnp.stack([x, y, z], axis=-1)


def get_spherical_grid(N, eps=0.0):
    lat = jnp.linspace(-90 + eps, 90 - eps, N // 2)
    lon = jnp.linspace(-180 + eps, 180 - eps, N)
    Lat, Lon = jnp.meshgrid(lat, lon)
    latlon_xs = jnp.concatenate([Lat.reshape(-1, 1), Lon.reshape(-1, 1)], axis=-1)
    spherical_xs = jnp.pi * (latlon_xs / 180.0) + jnp.array([jnp.pi / 2, jnp.pi])[None, :]
    xs = Hypersphere(2).spherical_to_extrinsic(spherical_xs)
    return xs, lat, lon


def plot_3d(x0, xt, size, prob=None):
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
        c = prob(xt) if prob is not None else "blue"
        cax = ax.scatter(x, y, z, s=50, vmin=0.0, vmax=2.0, c=c, cmap=cmap)
    # if grad is not None:
    #     u, v, w = grad[:, 0], grad[:, 1], grad[:, 2]
    #     quiver = ax.quiver(
    #         x, y, z, u, v, w, length=0.2, lw=2, normalize=False, cmap=cmap
    #     )
    #     quiver.set_array(c)

    plt.colorbar(cax)
    # plt.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return fig


def earth_plot(cfg, log_prob, train_ds, test_ds, N, azimuth=None, samples=None):
    """generate earth plots with model density or integral paths aka streamplot"""
    has_cartopy = importlib.find_loader("cartopy")
    print("has_cartopy", has_cartopy)
    if not has_cartopy:
        return

    # parameters
    azimuth_dict = {"earthquake": 70, "fire": 50, "floow": 60, "volcanoe": 170}
    azimuth = azimuth_dict[str(cfg.dataset.name)] if azimuth is None else azimuth
    polar = 30
    # projs = ["ortho", "robinson"]
    projs = ["ortho"]

    xs, lat, lon = get_spherical_grid(N, eps=0.0)
    # ts = [0.01, 0.05, cfg.flow.tf]
    ts = [cfg.flow.tf]
    figs = []

    for t in ts:
        print(t)
        # fs = log_prob(xs, t)
        fs = log_prob(xs)
        fs = fs.reshape((lat.shape[0], lon.shape[0]), order="F")
        fs = jnp.exp(fs)
        # norm = mcolors.PowerNorm(3.)  # NOTE: tweak that value
        norm = mcolors.PowerNorm(0.2)  # N=500
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

            vmin, vmax = 0.0, 1.0
            # n_levels = 900
            n_levels = 200
            levels = np.linspace(vmin, vmax, n_levels)
            cmap = sns.cubehelix_palette(
                light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
            )
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
            if samples is not None:
                samples = np.array(latlon_from_cartesian(samples)) * 180 / math.pi
                points = projection.transform_points(
                    ccrs.Geodetic(), samples[:, 1], samples[:, 0]
                )
                ax.scatter(points[:, 0], points[:, 1], s=1.0, c=[colors[1]], alpha=1.0)
            samples = train_ds.dataset.dataset.data
            samples = np.array(latlon_from_cartesian(samples)) * 180 / math.pi
            points = projection.transform_points(
                ccrs.Geodetic(), samples[:, 1], samples[:, 0]
            )
            ax.scatter(
                points[train_idx, 0],
                points[train_idx, 1],
                s=0.2,
                c=[colors[5]],
                alpha=0.2,
            )
            ax.scatter(
                points[test_idx, 0],
                points[test_idx, 1],
                s=0.2,
                c=[colors[0]],
                alpha=0.2,
            )
            # plt.close(fig)
            figs.append(fig)

    return figs


def plot_so3(x0, xt, size, **kwargs):
    colors = sns.color_palette("husl", 1)
    # colors = sns.color_palette("tab10")
    fig, axes = plt.subplots(
        2,
        3,
        # figsize=(1.2 * size, 0.6 * size),
        figsize=(2 * size, 1 * size),
        sharex=False,
        sharey="col",
        tight_layout=True,
    )
    # x_labels = [r"$\alpha$", r"$\beta$", r"$\gamma$"]
    x_labels = [r"$\phi$", r"$\theta$", r"$\psi$"]
    y_labels = ["Target", "Model"]
    # bins = round(math.sqrt(len(w[:, 0])))
    bins = 100

    # print(k, x0.shape, xt.shape)
    for i, x in enumerate([x0, xt]):
        if x is None:
            continue
        w = _SpecialOrthogonal3Vectors().tait_bryan_angles_from_matrix(x)
        # w = _SpecialOrthogonal3Vectors().rotation_vector_from_matrix(x)
        w = np.array(w)
        for j in range(3):
            axes[i, j].hist(
                w[:, j],
                bins=bins,
                density=True,
                alpha=0.3,
                color=colors[0],
                # label=f"Component #{k}",
            )
            if j == 1:
                axes[i, j].set(xlim=(-math.pi / 2, math.pi / 2))
                axes[i, j].set_xticks([-math.pi / 2, 0, math.pi / 2])
                axes[i, j].set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$"], color="k")
            else:
                axes[i, j].set(xlim=(-math.pi, math.pi))
                axes[i, j].set_xticks([-math.pi, 0, math.pi])
                axes[i, j].set_xticklabels([r"$-\pi$", "0", r"$\pi$"], color="k")
            if j == 0:
                axes[i, j].set_ylabel(y_labels[i], fontsize=30)
            # if i == 0 and j == 0:
            # axes[i, j].legend(loc="best", fontsize=20)
            if i == 0:
                axes[i, j].get_xaxis().set_visible(False)
            if i == 1:
                axes[i, j].set_xlabel(x_labels[j], fontsize=30)
            axes[i, j].tick_params(axis="both", which="major", labelsize=20)

    plt.close(fig)
    return fig


def proj_t2(x):
    return jnp.mod(
        jnp.stack(
            [jnp.arctan2(x[..., 0], x[..., 1]), jnp.arctan2(x[..., 2], x[..., 3])],
            axis=-1,
        ),
        jnp.pi * 2,
    )


def plot_t2(x0, xt, size, **kwargs):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.6 * size, 0.6 * size),
        sharex=False,
        sharey=False,
        tight_layout=True,
    )

    for i, x in enumerate([x0, xt]):
        x = proj_t2(x)
        axes[i].scatter(x[..., 0], x[..., 1], s=0.1)

    for ax in axes:
        ax.set_xlim([0, 2 * jnp.pi])
        ax.set_ylim([0, 2 * jnp.pi])
        ax.set_aspect("equal")

    plt.close(fig)
    return fig


import seaborn as sns


def plot_tn(x0, xt, size, **kwargs):
    n = xt.shape[-1]
    n = min(5, n // 4)

    fig, axes = plt.subplots(
        n,
        2,
        figsize=(0.6 * size, 0.6 * size * n / 2),
        sharex=False,
        sharey=False,
        tight_layout=True,
        squeeze=False,
    )
    # cmap = sns.mpl_palette("viridis")
    for i, x in enumerate([x0, xt]):
        if x is None:
            continue
        for j in range(n):
            x_ = proj_t2(x[..., (4 * j) : (4 * (j + 1))])
            axes[j][i].scatter(x_[..., 0], x_[..., 1], s=0.1)
            # sns.kdeplot(
            #     x=np.asarray(x_[..., 0]),
            #     y=np.asarray(x_[..., 1]),
            #     ax=axes[j][i],
            #     cmap=cmap,
            #     fill=True,
            #     # levels=15,
            # )

    axes = [item for sublist in axes for item in sublist]
    for ax in axes:
        ax.set_xlim([0, 2 * jnp.pi])
        ax.set_ylim([0, 2 * jnp.pi])
        ax.set_aspect("equal")

    plt.close(fig)
    return fig


def proj_t1(x):
    return jnp.mod(jnp.arctan2(x[..., 0], x[..., 1]), 2 * np.pi)


def plot_t1(x0, xt, size, **kwargs):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.6 * size, 0.6 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )

    for i, x in enumerate([x0, xt]):
        x = proj_t1(x)
        sns.kdeplot(x, ax=axes[i])
        plt.scatter(jnp.zeros_like(x), x, marker="|")

    for ax in axes:
        ax.set_xlim([0, 2 * jnp.pi])

    plt.close(fig)
    return fig


def plot_so3_uniform(x, size=10):
    colors = sns.color_palette("husl", len(x))
    fig, axes = plt.subplots(
        1,
        3,
        # figsize=(1.2 * size, 0.6 * size),
        figsize=(2 * size, 0.5 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )
    x_labels = [r"$\phi$", r"$\theta$", r"$\psi$"]
    # bins = round(math.sqrt(len(w[:, 0])))
    bins = 100
    w = _SpecialOrthogonal3Vectors().tait_bryan_angles_from_matrix(x)
    w = np.array(w)

    for j in range(3):
        if j == 1:
            grid = np.linspace(-np.pi / 2, np.pi / 2, 100)
            axes[j].set_xticks([grid[0], 0, grid[-1]])
            axes[j].set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$"], color="k")
            y = np.sin(grid + math.pi / 2) / 2
        else:
            grid = np.linspace(-np.pi, np.pi, 100, endpoint=True)
            axes[j].set_xticks([grid[0], 0, grid[-1]])
            axes[j].set_xticklabels([r"$-\pi$", "0", r"$\pi$"], color="k")
            y = 1 / (2 * np.pi) * np.ones_like(grid)
        axes[j].hist(
            w[:, j],
            bins=bins,
            density=True,
            alpha=0.3,
            color=colors[0],
            label=r"$x_{t_f}$",
        )
        axes[j].set(xlim=(grid[0], grid[-1]))
        axes[j].set_xlabel(x_labels[j], fontsize=30)
        axes[j].tick_params(axis="both", which="major", labelsize=20)
        axes[j].plot(grid, y, alpha=0.5, lw=4, color="black", label=r"$p_{ref}$")
        if j == 0:
            axes[j].legend(loc="best", fontsize=20)

    plt.close(fig)
    return fig


def plot_so3b(prob, lambda_x, N, size=10):
    fig, axes = plt.subplots(
        1,
        3,
        # figsize=(1.2 * size, 0.6 * size),
        figsize=(2 * size, 0.5 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )
    x_labels = [r"$\phi$", r"$\theta$", r"$\psi$"]
    prob = prob.reshape(N, N // 2, N)
    lambda_x = lambda_x.reshape(N, N // 2, N)

    for j in range(3):
        if j == 1:
            grid = np.linspace(-np.pi / 2, np.pi / 2, N // 2)
            axes[j].set_xticks([grid[0], 0, grid[-1]])
            axes[j].set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$"], color="k")
        else:
            grid = np.linspace(-np.pi, np.pi, N)
            axes[j].set_xticks([grid[0], 0, grid[-1]])
            axes[j].set_xticklabels([r"$-\pi$", "0", r"$\pi$"], color="k")

        y = jnp.mean(prob * lambda_x, axis=jnp.delete(jnp.arange(3), j))

        axes[j].set(xlim=(grid[0], grid[-1]))
        axes[j].set_xlabel(x_labels[j], fontsize=30)
        axes[j].tick_params(axis="both", which="major", labelsize=20)
        axes[j].plot(grid, y, alpha=0.5, lw=4, color="black", label=r"$p_{ref}$")
        if j == 0:
            axes[j].legend(loc="best", fontsize=20)

    plt.close(fig)
    return fig


def plot_normal(x, dim, size=10):
    colors = sns.color_palette("husl", len(x))
    fig, axes = plt.subplots(
        1,
        dim,
        # figsize=(1.2 * size, 0.6 * size),
        figsize=(2 * size, 0.5 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )
    bins = 100
    w = np.array(x)

    for j in range(w.shape[-1]):
        grid = np.linspace(-3, 3, 100)
        y = norm().pdf(grid)
        axes[j].hist(
            w[:, j],
            bins=bins,
            density=True,
            alpha=0.3,
            color=colors[0],
            label=r"$x_{t_f}$",
        )
        axes[j].set(xlim=(grid[0], grid[-1]))
        axes[j].set_xlabel(rf"$e_{j+1}$", fontsize=30)
        axes[j].tick_params(axis="both", which="major", labelsize=20)
        axes[j].plot(grid, y, alpha=0.5, lw=4, color="black", label=r"$p_{ref}$")
        if j == 0:
            axes[j].legend(loc="best", fontsize=20)

    plt.close(fig)
    return fig


def make_disk_grid(N, eps=1e-3, dim=2, radius=1.0):
    h = Hyperbolic(dim=dim, default_coords_type="ball")
    radius = radius - eps
    x = jnp.linspace(-radius, radius, N)
    xs = dim * [x]
    xs = jnp.meshgrid(*xs)
    xs = jnp.concatenate([x.reshape(-1, 1) for x in xs], axis=-1)
    mask = jnp.linalg.norm(xs, axis=-1) < 1.0 - eps
    lambda_x = h.metric.lambda_x(xs) ** 2 * mask.astype(float)
    volume = (2 * radius) ** dim

    return xs, volume, lambda_x


def plot_poincare(
    x0, xt, size, log_prob=None, coord_map=lambda x: x, grid_plot=False, **kwargs
):

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(0.6 * size, 0.6 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )
    cmap = sns.cubehelix_palette(
        light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
    )

    if grid_plot and log_prob is not None:
        N = 150
        xs, volume, lambda_x = make_disk_grid(N, eps=1e-2)

        idx = jnp.nonzero(lambda_x)[0]
        ys = xs[idx]
        lambda_x = lambda_x[idx]
        ys = coord_map(ys)

        prob = jnp.exp(log_prob(ys))
        # prob = jnp.exp(jax.vmap(log_prob)(ys[:, None, :]))
        print(f"{prob.min():.4f} | {prob.mean():.4f} | {prob.max():.4f}")
        idx_not_nan = jnp.nonzero(~jnp.isnan(prob))[0]
        nb = len(jnp.nonzero(jnp.isnan(prob))[0])
        tot = prob.shape[0]
        print(f"prop nan in prob: {nb / tot * 100:.1f}%")
        # prob = prob.at[jnp.isnan(prob) | jnp.isinf(prob)].set(0.0)
        # prob = jnp.exp(log_prob(xs))
        # prob = prob.at[jnp.isnan(prob)].set(0.0)
        # measure = prob * lambda_x

        measure = jnp.zeros((N * N)).at[idx].set(prob * lambda_x)
        Z = measure[idx_not_nan].mean() * volume
        print(f"Z={Z.item():.2f}")

        xs = xs.reshape(N, N, 2)
        measure = measure.reshape(N, N)
        ax.pcolormesh(
            xs[:, :, 0],
            xs[:, :, 1],
            measure,
            cmap=cmap,
            linewidth=0,
            rasterized=True,
            shading="gouraud",
        )

    if xt is not None:
        if log_prob is not None:
            prob = jnp.exp(log_prob(coord_map(xt)))
            nb = len(jnp.nonzero(jnp.isnan(prob))[0])
            tot = prob.shape[0]
            print(f"prop nan in prob: {nb / tot * 100:.1f}%")
            c = prob if nb == 0 else "orange"
        else:
            c = "blue"
        ax.scatter(xt[..., 0], xt[..., 1], alpha=0.3, s=2, c=c, label="model", cmap=cmap)
    if x0 is not None:
        ax.scatter(
            x0[..., 0], x0[..., 1], alpha=1.0, s=2, c="black", label="data", cmap=cmap
        )

    # if xt is not None or x0 is not None:
    #     ax.legend(loc="best", fontsize=20)

    ax.set_xlim([-1.01, 1.01])
    ax.set_ylim([-1.01, 1.01])

    ax.add_patch(Circle((0, 0), 1.0, color="black", fill=False, linewidth=2, zorder=10))
    ax.set_aspect("equal")
    ax.axis("off")

    plt.close(fig)
    return fig


def plot(manifold, x0, xt, log_prob=None, size=10):
    prob = None if log_prob is None else lambda x: jnp.exp(log_prob(x))
    if isinstance(manifold, Euclidean) and manifold.dim == 3:
        fig = plot_3d(x0, xt, size, prob=prob)
    elif isinstance(manifold, Hypersphere) and manifold.dim == 2:
        fig = plot_3d(x0, xt, size, prob=prob)
    elif isinstance(manifold, _SpecialOrthogonalMatrices) and manifold.dim == 3:
        fig = plot_so3(x0, xt, size, prob=prob)
    elif (
        isinstance(manifold, ProductSameManifold)
        and isinstance(manifold.manifold, Hypersphere)
        and manifold.manifold.dim == 1
        and manifold.dim == 1
    ) or (isinstance(manifold, Hypersphere) and manifold.dim == 1):
        fig = plot_t1(x0, xt, size, prob=prob)
    elif (
        isinstance(manifold, ProductSameManifold)
        and isinstance(manifold.manifold, Hypersphere)
        and manifold.manifold.dim == 1
    ):
        fig = plot_tn(x0, xt, size, prob=prob)
    elif isinstance(manifold, PoincareBall) and manifold.dim == 2:
        fig = plot_poincare(x0, xt, size, log_prob=log_prob)
    elif isinstance(manifold, Hyperboloid) and manifold.dim == 2:
        coord_map = Hyperbolic._ball_to_extrinsic_coordinates
        if x0 is not None:
            x0 = Hyperbolic._extrinsic_to_ball_coordinates(x0)
        if xt is not None:
            xt = Hyperbolic._extrinsic_to_ball_coordinates(xt)
        fig = plot_poincare(x0, xt, size, coord_map=coord_map, log_prob=log_prob)
    else:
        print("Only plotting over R^3, S^2, S1/T1, T2, TN, H2 and SO(3) is implemented.")
        return None
    return fig


def plot_ref(manifold, xt, size=10, log_prob=None):
    if isinstance(manifold, Euclidean):
        fig = plot_normal(xt, manifold.dim, size)
    elif isinstance(manifold, Hypersphere) and manifold.dim == 2:
        fig = None
    elif isinstance(manifold, _SpecialOrthogonalMatrices) and manifold.dim == 3:
        fig = plot_so3_uniform(xt, size)
    elif isinstance(manifold, PoincareBall) and manifold.dim == 2:
        fig = plot_poincare(xt, None, size, log_prob=log_prob)
    elif isinstance(manifold, Hyperboloid) and manifold.dim == 2:
        coord_map = Hyperbolic._ball_to_extrinsic_coordinates
        if xt is not None:
            xt = Hyperbolic._extrinsic_to_ball_coordinates(xt)
        fig = plot_poincare(
            xt, None, size, log_prob=log_prob, coord_map=coord_map, grid_plot=True
        )
    else:
        print("Only plotting over R^3, S^2 and SO(3) is implemented.")
        return None
    return fig
