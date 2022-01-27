import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # s
import matplotlib.colors as mcolors
import logging
import torch
import os

try:
    plt.switch_backend("MACOSX")
except ImportError as error:
    plt.switch_backend("agg")
import seaborn as sns

from geoflow.distributions import Empirical
from geoflow.flows import TrivialFlow
from .utils import *

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


def cartesian_from_latlon(x):
    """ Embedded 3D unit vector from spherical polar coordinates.

    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians.

    Returns
    -------
    nhat : numpy.array
        unit vector(s) in direction (phi, theta).
    """
    assert x.shape[-1] == 2
    lat = x.select(-1, 0)
    lon = x.select(-1, 1)
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)


def latlon_from_cartesian(x):
    r = x.pow(2).sum(-1).sqrt()
    y = x.select(-1, 1)
    z = x.select(-1, 2)
    x = x.select(-1, 0)

    lat = torch.asin(z / r)
    lon = torch.fmod(torch.atan2(y, x), math.pi * 2)
    return torch.cat([lat.unsqueeze(-1), lon.unsqueeze(-1)], dim=-1)


def partial_latlon_unit_vectors(x):
    lat = x[:, [0]]
    lon = x[:, [1]]
    dlat = torch.cat([-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()], dim=-1)  # unit vector (by default)
    dlon = torch.cat([-lon.sin(), lon.cos(), torch.zeros_like(lon)], dim=-1)  # unit vector (normalized)
    B = torch.cat([dlat.unsqueeze(1), dlon.unsqueeze(1)], dim=1)
    return B


def pt_plot(args, model, target, manifold, device, filepath, N=100, final=False):
    # Create gif based on saved images
    dirpath = "/".join(filepath.split("/")[:-1])
    pts = []
    files = os.listdir(dirpath)
    files.sort()
    for file in files:
        if file.endswith(".pt"):
            pt = torch.load(file, map_location=torch.device("cpu"))
            print(pt.mean().item())
            pts.append(pt)
            # os.remove(img_file)  # remove img_file as we no longer need it
    print(len(pts))
    raise


def gif_plot(args, model, target, manifold, device, filepath, N=100, final=False):
    # Create gif based on saved images
    import imageio

    dirpath = "/".join(filepath.split("/")[:-1])
    imgs = []
    files = os.listdir(dirpath)
    files.sort()
    i = 0
    for file in files:
        if file.endswith((".png", ".jpg")):
            if len(imgs) >= 50:
                break
            if i % 1 == 0:
                print(os.path.join(dirpath, file))
                imgs.append(imageio.imread(file))
                # os.remove(img_file)  # remove img_file as we no longer need it
            i += 1
    filename = os.path.join(dirpath, "density.gif")
    print(len(imgs))
    imageio.mimwrite(filename, imgs)


def compute_microbatch_split(x, K=1):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x.size(0)
    S = int(2e5 / (K * np.prod(x.size()[1:])))  # float heuristic for 12Gb cuda memory
    assert S > 0, "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def _compute_prob_vec(dist, x):
    return dist.log_prob(x).exp()


def compute_prob(dist, x):
    S = compute_microbatch_split(x)
    lw = torch.cat([_compute_prob_vec(dist, _x.contiguous()) for _x in x.split(S)], 0)  # concat on batch
    return lw


def compute_or_reload(compute_fn, path, force=False):
    if not force and os.path.isfile(path):
        print("load {}".format(path))
        tensor = torch.load(path, map_location=torch.device("cpu"))
    else:
        tensor = compute_fn()
        if not force:
            print("save {}".format(path))
            torch.save(tensor, path)
    return tensor


def get_doc_path_for_plot_earth(args, filepath, proj, flow):
    """ if plot is included in latex get proper path for it """
    target_names = {"Fire": "Fire", "Flood": "Flood", "QuakesBig": "Earthquake", "Volerup": "Volcano"}
    target_name = target_names[str(args.target)].lower()
    exp_dir_idx = filepath.split("/").index("experiments")
    path = "/".join(filepath.split("/")[:exp_dir_idx]) + "/doc/images/earth/densities2"
    type_plot = "flow" if flow else "density"
    doc_fig_name = "{}/{}_{}_{}_{}".format(path, type_plot, proj, target_name, str(args.model))
    return doc_fig_name


def earth_density_plot(args, model, target, manifold, device, filepath, N, final):
    return earth_plot(args, model, target, manifold, device, filepath, flow=False, azimuth=None, N=N, final=final)


def earth_flow_plot(args, model, target, manifold, device, filepath, N, final):
    return earth_plot(args, model, target, manifold, device, filepath, flow=True, azimuth=100, N=N, final=final)


def earth_plot(args, model, target, manifold, device, filepath, flow, azimuth, N, final):
    """ generate earth plots with model density or integral paths aka streamplot  """
    cmap = sns.cubehelix_palette(light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True)

    # parameters
    azimuth_dict = {"QuakesBig": 70, "Fire": 50, "Flood": 60, "Volerup": 170}
    azimuth = azimuth_dict[str(args.target)] if azimuth is None else azimuth
    polar = 30
    projs = ["ortho"] if not flow else ["ortho"]

    if issubclass(type(target), Empirical):
        samples = target.data.to(device)
        samples = latlon_from_cartesian(samples).data.cpu().numpy() * 180 / math.pi

    # build spherical (lat, lon) grid and compute pdf
    eps = 1e-3
    N_lon, N_lat = N, N
    if flow:
        # /!\ points need to not be located outside the visible part of the orthographic projection
        polar = 0
        lon = torch.linspace(-90 + azimuth, 90 + azimuth, N_lon).to(device)
    else:
        lon = torch.linspace(-180, 180, N_lon).to(device)
    lat = torch.linspace(-90 + eps, 90 - eps, N_lat).to(device)
    latlat, lonlon = torch.meshgrid([lat, lon])
    xs_latlon = torch.cat([latlat.reshape(-1, 1), lonlon.reshape(-1, 1)], dim=1)
    xs = cartesian_from_latlon(xs_latlon / 180 * math.pi)

    if not flow:
        grid_pdf_path = "{}_grid_{}.pt".format(filepath, N)
        fs = compute_or_reload(lambda: compute_prob(model, xs).view(N_lat, N_lon), grid_pdf_path, force=not save_grid)
        fs = fs.detach().cpu().numpy()

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

        # plot vector field
        if flow:
            # earth features
            ax.add_feature(cfeature.LAND, zorder=0, facecolor="#cdcdcd")

            # get vecor field grid discretisation
            T = torch.tensor([0.5]).to(device)
            diffeq = model.flow[0].odefunc.diffeq
            dy = diffeq(T, xs)
            x, y = lonlon.cpu().numpy(), latlat.cpu().numpy()

            # project on polar coordinates
            # https://github.com/SciTools/cartopy/issues/1179, https://stackoverflow.com/questions/50454322/matplotlib-cartopy-streamplot-results-in-qhullerror-with-some-projections
            xs_latlon = latlon_from_cartesian(xs)
            b_latlon = partial_latlon_unit_vectors(xs_latlon)
            dy_latlon = torch.matmul(b_latlon, dy.unsqueeze(-1)).squeeze(-1).view(N_lon, N_lat, 2)
            u, v = dy_latlon[:, :, 1], dy_latlon[:, :, 0]
            u, v = u.cpu().numpy(), v.cpu().numpy()

            u_src_crs, v_src_crs = u / np.cos(latlat.numpy() / 180 * math.pi), v
            magnitude = np.sqrt(u ** 2 + v ** 2)
            magn_src_crs = np.sqrt(u_src_crs ** 2 + v_src_crs ** 2)
            u_src_crs *= magnitude / magn_src_crs
            v_src_crs *= magnitude / magn_src_crs
            u, v = u_src_crs, v_src_crs
            vector_crs = ccrs.PlateCarree()

            # plot vector field
            # ax.set_extent((-180, 180, 50, 90), vector_crs)
            # ax.quiver(x, y, u, v, transform=vector_crs, angles="xy")
            # fig_name = "{}_quiverplot_{}_{}_{}".format(filepath, proj, azimuth, N)
            # plt.savefig(fig_name + ".png", dpi=300, bbox_inches="tight", transparent=True)

            magnitude = (u ** 2 + v ** 2) ** 0.5
            lw = 4  #  5 * magnitude / magnitude.max()

            # plot and save
            ax.streamplot(x, y, u, v, transform=vector_crs, linewidth=lw, density=1.4, color=magnitude, cmap=cmap)
            fig_name = "{}_streamplot_{}_{}_{}".format(filepath, proj, azimuth, N)
            # plt.savefig(fig_name + ".png", dpi=dpi, bbox_inches="tight", transparent=True)
            # fig_name = get_doc_path_for_plot_earth(args, filepath, proj, flow) if final else fig_name
            # plt.savefig(fig_name + ".jpg", dpi=dpi, quality=80, bbox_inches="tight")
            plt.clf()

        else:
            # earth features
            # ax.add_feature(cfeature.LAND, zorder=0, facecolor="#595959")
            ax.add_feature(cfeature.LAND, zorder=0, facecolor="#e0e0e0")

            extent = (-180, 180, -90, 90)
            # ax.contourf(lon, lat, fs, levels=900, alpha=0.8, vmin=0.0, vmax=2.0, transform=ccrs.PlateCarree(), antialiased=True, cmap=cmap)
            norm = mcolors.PowerNorm(0.5)
            # norm = mcolors.Normalize(vmin=np.min(fs), vmax=np.max(fs))
            print(fs.min().item(), np.median(fs).item(), fs.mean().item(), fs.max().item())
            fs = np.array(fs)
            # fs = np.ma.masked_where(fs <= 0.1, fs)
            fs = norm(fs)
            # fs = fs / fs.mean() * 0.018
            print(fs.min().item(), np.median(fs).item(), fs.mean().item(), fs.max().item())
            # fs[fs <= 0.1] = -100.
            from matplotlib import cm
            from matplotlib.colors import ListedColormap

            # cmap = cm.get_cmap("viridis")
            # cmap = sns.cubehelix_palette(light=1, as_cmap=True)

            # my_cmap = cmap(np.arange(cmap.N))
            # my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
            # my_cmap = ListedColormap(my_cmap)
            # print(my_cmap[:, -1])

            # viridis = cm.get_cmap("viridis", 256)
            # newcolors = cmap(np.linspace(0, 1, 256))
            # newcolors[:, -1] = 1.0
            # my_cmap = ListedColormap(newcolors)
            # print(my_cmap(fs))
            # def my_cmap(x):
            #     res = cmap(x)
            #     res[-1] = 0.5
            #     return res

            # my_cmap = ListedColormap(my_cmap)
            # cmap = cm.get_cmap("RdPu")
            vmin, vmax = 0.2, 1.0
            levels = np.linspace(vmin, vmax, 900)
            colors = cmap(np.linspace(vmin, vmax, 900))
            colors[:, -1] = 0.0
            # print(colors)
            # raise
            fs = 1.0 * fs
            cs = ax.contourf(
                lon,
                lat,
                fs,
                levels=levels,
                alpha=0.8,
                transform=ccrs.PlateCarree(),
                antialiased=True,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                # colors=colors
                extend="both",
            )
            cs.cmap.set_over(colors[-1, :])
            cs.cmap.set_under("blue")
            for col in ax.collections[0:1]:
                col.set_alpha(0)
            # ax.collections[10].set_alpha(0)

            # add scatter plots of the dataset
            if issubclass(type(target), Empirical):
                colors = sns.color_palette("hls", 2)
                points = projection.transform_points(ccrs.Geodetic(), samples[:, 1], samples[:, 0])
                ax.scatter(
                    points[target.train_dataset.indices, 0], points[target.train_dataset.indices, 1], s=1.0 / 2, c=[colors[1]], alpha=0.5
                )
                ax.scatter(
                    points[target.test_dataset.indices, 0], points[target.test_dataset.indices, 1], s=1.0 / 2, c=[colors[0]], alpha=0.5
                )

            # save plot as file
            fig_name = "{}_{}_{}_{}".format(filepath, proj, azimuth, N)
            # plt.savefig(fig_name + ".png", dpi=dpi, bbox_inches="tight", transparent=True)
            # fig_name = get_doc_path_for_plot_earth(args, filepath, proj, flow) if final else fig_name
            plt.savefig(fig_name + ".jpg", dpi=dpi, quality=80, bbox_inches="tight")
            plt.clf()
    return fig


def get_doc_path_for_plot_sphere(args, filepath, obj):
    """ if plot is included in latex get proper path for it """
    exp_dir_idx = filepath.split("/").index("experiments")
    path = "/".join(filepath.split("/")[:exp_dir_idx]) + "/doc/images/vmf/densities2"
    target_name = args.target_param_scale
    doc_fig_name = "{}/{}_{}_{}".format(path, obj, target_name, str(args.model))
    return doc_fig_name


def vmf_plot(args, model, target, manifold, device, filepath, N=100, final=False):

    fig = plt.figure(figsize=(7.5, 5))
    projection = "3d"
    cmap = sns.cubehelix_palette(light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True)

    eps = 0  # 1e-3
    lon = torch.linspace(-180 + eps, 180 - eps, N).to(device)
    lat = torch.linspace(-90 + eps, 90 - eps, N // 2).to(device)

    X, Y = torch.meshgrid([lat, lon])
    xs_latlon = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
    xs = cartesian_from_latlon(xs_latlon / 180 * math.pi)
    manifold.assert_check_point_on_manifold(xs)

    lambda_x = torch.cos(xs_latlon[:, 0] / 180 * math.pi).view(N // 2 * N, 1)

    prior = copy.deepcopy(model)
    prior.flow = TrivialFlow()
    models = [model, target]

    for i, mod in enumerate(models):

        ax = fig.add_subplot(1, len(models), i + 1, projection=projection)
        sphere_plot(manifold, ax)
        ax.set_aspect("equal")
        ax.set_axis_off()

        grid_pdf_path = "{}_grid_{}.pt".format(filepath, N)
        pdf = compute_or_reload(lambda: mod.log_prob(xs).exp().view(N // 2 * N, 1), grid_pdf_path, mod != model or not save_grid)
        nu = pdf * lambda_x
        volume = (2 * np.pi) * np.pi
        Z = (nu * volume).mean()
        log.info("volume = {:.2f}".format(Z.item()))

        ax.view_init(elev=30, azim=180)
        xs2 = xs.detach().view(N // 2, N, manifold.coord_dim).cpu().numpy()
        fs = pdf.view(N // 2, N).cpu().numpy()

        x, y, z = xs2[:, :, 0], xs2[:, :, 1], xs2[:, :, 2]
        ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, rasterized=True, facecolors=cmap(fs))

    fig.tight_layout(pad=-18.0)
    fig_name = "{}_{}_{}".format(filepath, args.obj, N)
    plt.savefig(fig_name + ".png", dpi=300, bbox_inches="tight", transparent=True, pad_inches=-1.2)
    fig_name = get_doc_path_for_plot_sphere(args, filepath, args.obj) if final else fig_name
    plt.savefig(fig_name + ".jpg", dpi=300, quality=90, bbox_inches="tight", transparent=True, pad_inches=-1.2)
    return fig

