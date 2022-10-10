import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere

import jax
from jax import numpy as jnp
import numpy as np

from score_sde.utils import batch_mul
from score_sde.datasets import vMFDataset
from riemannian_score_sde.sde import Brownian
from riemannian_score_sde.utils.vis import (
    remove_background,
    sphere_plot,
    get_sphere_coords,
)
from scripts.utils import (
    plot_and_save2,
    plot_and_save_video,
    plot_and_save_video2,
    vMF_pdf,
)


def plot(traj, size=10, dpi=300, out="out", color="red"):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=0, azim=0)
    sphere = visualization.Sphere()
    sphere_color = (220 / 255, 220 / 255, 220 / 255)
    ax = sphere_plot(ax, color=sphere_color)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax = remove_background(ax)

    N, K, D = traj.shape
    colours = sns.color_palette("hls", K)
    colours = sns.cubehelix_palette(n_colors=K, reverse=False)
    alpha = np.linspace(0, 1, N + 1)
    # alpha = np.flip(alpha)
    ax.scatter(
        traj[-1, :, 0],
        traj[-1, :, 1],
        traj[-1, :, 2],
        s=100,
        marker="*",
        color=colours[0],
    )
    for k in range(K):
        c = sns.cubehelix_palette(n_colors=N, reverse=False, as_cmap=False)
        for n in range(N - 1):
            ax.plot(
                traj[n : n + 2, k, 0],
                traj[n : n + 2, k, 1],
                traj[n : n + 2, k, 2],
                lw=1.0,
                linestyle="-",
                # alpha = alpha[n],
                alpha=0.9,
                # color=colours[k])
                color=c[n],
            )
    for i in range(1):
        ax.scatter(
            traj[-1, :, 0],
            traj[-1, :, 1],
            traj[-1, :, 2],
            s=100,
            marker="o",
            alpha=1,
            color=colours[-1],
        )
    print("save")
    fig.tight_layout()
    plt.savefig(
        out + ".jpg", dpi=dpi, bbox_inches="tight", transparent=True, pad_inches=-2.1
    )
    plt.close(fig)
    return fig


def make_animation(traj, size=20, dpi=300, fps=10, out="out"):
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=fps)
    ratio = size / 10
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=0, azim=0)
    sphere_color = (220 / 255, 220 / 255, 220 / 255)
    ax = sphere_plot(ax, color=sphere_color)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax = remove_background(ax)
    fig.subplots_adjust(left=-0.45, bottom=-0.45, right=1.4, top=1.4, wspace=0, hspace=0)

    N, K, D = traj.shape
    colours = sns.color_palette("hls", K)
    colours = sns.cubehelix_palette(n_colors=K, reverse=False)
    c = sns.cubehelix_palette(n_colors=N, reverse=False, as_cmap=False)
    # alpha = np.linspace(0, 1, N+1)
    # alpha = np.flip(alpha)
    with writer.saving(fig, out, dpi=dpi):
        ax.scatter(
            traj[0, :, 0],
            traj[0, :, 1],
            traj[0, :, 2],
            s=ratio * 500,
            marker="X",
            color=colours[0],
        )
        writer.grab_frame()
        for n in range(N - 1):
            for k in range(K):
                ax.plot(
                    traj[n : n + 2, k, 0],
                    traj[n : n + 2, k, 1],
                    traj[n : n + 2, k, 2],
                    lw=ratio * 1.0,
                    linestyle="-",
                    # alpha = alpha[n],
                    alpha=0.9,
                    # color=colours[k])
                    color=c[n],
                )
            writer.grab_frame()
        ax.scatter(
            traj[-1, :, 0],
            traj[-1, :, 1],
            traj[-1, :, 2],
            s=ratio * 500,
            marker="o",
            alpha=1,
            color=colours[-1],
        )
        writer.grab_frame()
        for _ in range(20):
            writer.grab_frame()
        writer.finish()
    print("save")


def make_animation_2(traj, kernel_widths, kernel, size=20, dpi=300, fps=10, out="out"):
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=fps)
    ratio = size / 10
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=0, azim=0)
    sphere_color = (220 / 255, 220 / 255, 220 / 255)
    # ax = sphere_plot(ax, color=sphere_color)
    x, y, z = get_sphere_coords()
    xs = jnp.stack([x, y, z], axis=-1)

    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax = remove_background(ax)
    fig.subplots_adjust(left=-0.45, bottom=-0.45, right=1.4, top=1.4, wspace=0, hspace=0)

    N, K, D = traj.shape
    colours = sns.color_palette("hls", K)
    colours = sns.cubehelix_palette(n_colors=K, reverse=False)
    # alpha = np.linspace(0, 1, N+1)
    # alpha = np.flip(alpha)
    c = sns.cubehelix_palette(n_colors=N, reverse=False, as_cmap=False)
    cmap_pdf = sns.cubehelix_palette(
        light=1, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
    )

    with writer.saving(fig, out, dpi=dpi):
        fs = kernel(xs, kernel_widths[0])
        sphere = ax.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            cmap=cmap_pdf,
            linewidth=0,
            antialiased=False,
            rasterized=True,
            facecolors=cmap_pdf(fs),
        )
        ax.scatter(
            traj[0, :, 0],
            traj[0, :, 1],
            traj[0, :, 2],
            s=ratio * 500,
            marker="X",
            color=colours[0],
        )
        writer.grab_frame()
        for n in range(N - 1):
            for k in range(K):
                sphere.set_facecolors(
                    cmap_pdf(kernel(xs, kernel_widths[n])).reshape((-1, 4))
                )
                ax.plot(
                    traj[n : n + 2, k, 0],
                    traj[n : n + 2, k, 1],
                    traj[n : n + 2, k, 2],
                    lw=ratio * 1.0,
                    linestyle="-",
                    # alpha = alpha[n],
                    alpha=0.9,
                    # color=colours[k])
                    color=c[n],
                )
            writer.grab_frame()
        sphere.set_facecolors(cmap_pdf(kernel(xs, kernel_widths[-1])).reshape((-1, 4)))
        ax.scatter(
            traj[-1, :, 0],
            traj[-1, :, 1],
            traj[-1, :, 2],
            s=ratio * 500,
            marker="o",
            alpha=1,
            color=colours[-1],
        )
        writer.grab_frame()
        for _ in range(20):
            writer.grab_frame()
        writer.finish()
    print("save")


@partial(jax.jit, static_argnums=(5))
def brownian_motion_traj(previous_x, N, dt, timesteps, traj, sde):
    rng = jax.random.PRNGKey(2)

    def body(step, val):
        rng, x, traj = val
        traj = traj.at[step].set(x)
        t = jnp.broadcast_to(timesteps[step], (x.shape[0], 1))
        rng, z = sde.manifold.random_normal_tangent(
            state=rng, base_point=x, n_samples=x.shape[0]
        )
        drift, diffusion = sde.coefficients(x, t)  # sde.sde(x, t)
        tangent_vector = drift * dt + batch_mul(diffusion, jnp.sqrt(dt) * z)
        x = sde.manifold.metric.exp(tangent_vec=tangent_vector, base_point=x)
        return rng, x, traj

    _, x, traj = jax.lax.fori_loop(0, N, body, (rng, previous_x, traj))
    traj = traj.at[-1].set(x)
    return traj


def test_heat_kernel():
    dataset = vMFDataset([K], jax.random.PRNGKey(0), S2, mu=mu.reshape(3), kappa=kappa)


def plot_and_save_frame(
    traj, pdf=None, size=20, dpi=300, out="out.jpg", color="red", N=None
):
    ratio = size / 10
    fig = plt.figure(figsize=(size, size))
    # ax = fig.add_subplot(111, projection="3d")
    ax = Axes3D(fig, computed_zorder=False)
    ax.view_init(elev=0, azim=0)
    sphere_color = (220 / 255, 220 / 255, 220 / 255)
    # ax = sphere_plot(ax, color=sphere_color)
    x, y, z = get_sphere_coords()
    xs = jnp.stack([x, y, z], axis=-1)

    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax = remove_background(ax)
    fig.subplots_adjust(left=-0.45, bottom=-0.45, right=1.4, top=1.4, wspace=0, hspace=0)

    if N is None:
        N, K, D = traj.shape
    else:
        _, K, D = traj.shape
    colours = sns.color_palette("hls", K)
    colours = sns.cubehelix_palette(n_colors=K, reverse=False)
    # alpha = np.linspace(0, 1, N+1)
    # alpha = np.flip(alpha)
    c = sns.cubehelix_palette(n_colors=N, reverse=False, as_cmap=False)
    cmap_pdf = sns.cubehelix_palette(
        light=0.95, dark=0.05, start=0.5, rot=-0.75, reverse=False, as_cmap=True
    )

    fs = pdf(xs)
    sphere = ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        cmap=cmap_pdf,
        linewidth=0,
        antialiased=False,
        rasterized=True,
        facecolors=cmap_pdf(fs),
        zorder=0,
        alpha=1.0,
    )
    ax.scatter(
        traj[0, :, 0],
        traj[0, :, 1],
        traj[0, :, 2],
        s=ratio * 200,
        marker="X",
        color=c[0],
    )

    for n in range(N - 1):
        for k in range(K):
            ax.plot(
                traj[n : n + 2, k, 0] * 1.01,
                traj[n : n + 2, k, 1] * 1.01,
                traj[n : n + 2, k, 2] * 1.01,
                lw=ratio * 2.0,
                linestyle="-",
                # alpha = alpha[n],
                alpha=0.9,
                # color=colours[k])
                color=c[n],
                zorder=1,
            )
    # if traj.shape[0] == N:
    ax.scatter(
        traj[-1, :, 0],
        traj[-1, :, 1],
        traj[-1, :, 2],
        s=ratio * 200,
        marker="o",
        color=c[traj.shape[0] - 1],
        edgecolor=None,
    )

    plt.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def test_s2():
    print("test_s2")
    S2 = Hypersphere(dim=2)
    sde = Brownian(S2, tf=1, beta_0=0.0001, beta_f=2)
    # sde = Brownian(S2, tf=4, beta_0=1, beta_f=1)

    K = 4
    mu = jnp.array([[1.0, 0.0, 0.0]])
    kappa = 200
    dataset = vMFDataset([K], jax.random.PRNGKey(0), S2, mu=mu.reshape(3), kappa=kappa)

    N = 300
    eps = 1e-3
    tf = sde.tf
    # tf = 0.5
    dt = (tf - eps) / N
    timesteps = jnp.linspace(eps, tf, N)
    previous_x = next(dataset)
    traj = jnp.zeros((N + 1, previous_x.shape[0], S2.dim + 1))
    traj = brownian_motion_traj(previous_x, N, dt, timesteps, traj, sde)
    # traj = previous_x
    print(traj.shape)
    print("plot")
    traj = jnp.flip(traj, axis=0)
    # plot(traj, out="images/traj_plot")

    def kernel(x, t):
        x_shape = x.shape[:-1]
        x = x.reshape((-1, 3))

        _, std = sde.marginal_prob(x, 1 - t)

        t_ = 1 - t
        s = -2 * (-0.25 * t_**2 * (sde.beta_f - sde.beta_0) - 0.5 * t_ * sde.beta_0)

        p = np.exp(
            S2.log_heat_kernel(
                jnp.repeat(jnp.array([1, 0, 0])[None, :], x.shape[0], axis=0),
                x,
                s + 1 / 200,
                0.1,
                51,
            )
        )
        return p.reshape(*x_shape)

    # make_animation_2(
    #     traj,
    #     jnp.linspace(0.5, 1, traj.shape[0]),
    #     kernel,
    #     dpi=100,
    #     size=10,
    #     fps=100,
    #     out="images/traj_plot.mp4",
    # )

    steps = N
    # steps = 3
    for i, t in zip(jnp.linspace(1, N, steps), jnp.linspace(0, 1, steps)):
        print(t)
        i = int(i)
        plot_and_save_frame(
            traj[:i, :, :],
            partial(kernel, t=t),
            out=f"/home/mhutchin/Documents/projects/score-sde/images/gif2/frame_{t:0.4f}.png",
            N=N,
            size=10,
            dpi=50,
        )

    # for i in range(N):
    #     plot_and_save2(trajectories)
    # plot(traj, out="images/traj_plot.png")
    print("end")


if __name__ == "__main__":
    test_s2()
