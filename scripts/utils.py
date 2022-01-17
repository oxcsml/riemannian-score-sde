import os
import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns

import geomstats.backend as gs
import geomstats.visualization as visualization

import jax
from jax import numpy as jnp
import numpy as np


def plot_and_save_video(
    trajectories, pdf=None, size=20, fps=10, dpi=100, out="out.mp4", color="red"
):
    """Render a set of geodesics and save it to an mpeg 4 file."""
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    sphere = visualization.Sphere()
    if pdf:
        scatter = sphere.plot_heatmap(ax, pdf)
    points = gs.to_ndarray(trajectories[0], to_ndim=2)
    sphere.draw(ax, color=color, marker=".")
    scatter = sphere.draw_points(ax, points=points, color=color, marker=".")
    with writer.saving(fig, out, dpi=dpi):
        for i, points in enumerate(trajectories[1:]):
            print(i, len(trajectories))
            points = gs.to_ndarray(points, to_ndim=2)
            scatter.remove()
            scatter = sphere.draw_points(ax, points=points, color=color, marker=".")
            writer.grab_frame()


def plot_and_save_video2(
    heatmaps, size=20, fps=10, dpi=100, out="out.mp4", color="red"
):
    """Render a set of geodesics and save it to an mpeg 4 file."""
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=45)
    cmap = sns.cubehelix_palette(light=1, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True)
    sphere = visualization.Sphere()
    sphere.draw(ax, color=color, marker=".")
    N = 100
    eps = 1e-3
    theta = jnp.linspace(eps, jnp.pi - eps, N // 2)
    phi = jnp.linspace(eps, 2 * jnp.pi - eps, N)
    theta, phi = jnp.meshgrid(theta, phi)
    theta = theta.reshape(-1, 1)
    phi = phi.reshape(-1, 1)
    xs = jnp.concatenate([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi), 
        jnp.cos(theta)
    ], axis=-1)
    
    with writer.saving(fig, out, dpi=dpi):
        for i, heatmap in enumerate(heatmaps):
            print(i, len(heatmaps))
            fs = heatmap(xs)
            xs_reshaped = xs.reshape(N // 2, N, 3)
            fs = fs.reshape(N // 2, N)
            surf = ax.plot_surface(
                    xs_reshaped[:, :, 0],
                    xs_reshaped[:, :, 1],
                    xs_reshaped[:, :, 2],
                    rstride=1,
                    cstride=1,
                    cmap=cmap,
                    linewidth=0,
                    antialiased=False,
                    rasterized=True,
                    facecolors=cmap(fs),
                )
            writer.grab_frame()
            surf.remove()


def plot_and_save(
    trajectories, pdf=None, size=20, dpi=300, out="out.jpg", color="red"
):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=45)
    # colours = sns.cubehelix_palette(n_colors=trajectories.shape[0], light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False)
    colours = ["green", "blue"]
    sphere = visualization.Sphere()
    cmap = sns.cubehelix_palette(light=1, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True)
    if pdf:
        sphere.plot_heatmap(ax, pdf, n_points=16000, alpha=0.2, cmap=cmap)
    sphere.draw(ax, color=color, marker=".")
    for i, points in enumerate(trajectories):
        points = gs.to_ndarray(points, to_ndim=2)
        sphere.draw_points(ax, points=points, color=colours[i], marker=".")
    plt.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)

def plot_and_save2(
    trajectories, pdf=None, size=20, dpi=300, out="out.jpg", color="red"
):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=45)
    # colours = sns.cubehelix_palette(n_colors=trajectories.shape[0], light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False)
    colours = ["green", "blue"]
    sphere = visualization.Sphere()
    cmap = sns.cubehelix_palette(light=1, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True)
    # if pdf:
    #     sphere.plot_heatmap(ax, pdf, n_points=16000, alpha=0.2, cmap=cmap)
    N = 200
    eps = 1e-3
    theta = jnp.linspace(eps, jnp.pi - eps, N // 2)
    phi = jnp.linspace(eps, 2 * jnp.pi - eps, N)
    theta, phi = jnp.meshgrid(theta, phi)
    theta = theta.reshape(-1, 1)
    phi = phi.reshape(-1, 1)
    xs = jnp.concatenate([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi), 
        jnp.cos(theta)
    ], axis=-1)
    fs = pdf(xs)
    xs = xs.reshape(N // 2, N, 3)
    fs = fs.reshape(N // 2, N)
    ax.plot_surface(
            xs[:, :, 0],
            xs[:, :, 1],
            xs[:, :, 2],
            rstride=1,
            cstride=1,
            cmap=cmap,
            linewidth=0,
            antialiased=False,
            rasterized=True,
            facecolors=cmap(fs),
        )
    sphere.draw(ax, color=color, marker=".")
    plt.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)



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


def plot_and_save3(
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


def plot_and_save_video3(
    traj_x, traj_f, traj_grad_f, timesteps, size=20, fps=10, dpi=100, out="out.mp4", color="red"
):
    """Render a set of geodesics and save it to an mpeg 4 file."""
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax = remove_background(ax)
    fig.subplots_adjust(left=-0.2, bottom=-0.2, right=1.2, top=1.2, wspace=0, hspace=0)
    # fig.subplots_adjust(wspace=-100, hspace=100)
    ax.view_init(elev=30, azim=45)
    colours = sns.cubehelix_palette(n_colors=traj_x.shape[1])
    cmap = sns.cubehelix_palette(as_cmap=True)
    sphere = visualization.Sphere()
    sphere.draw(ax, color=color, marker=".")
    n = traj_x.shape[0]
    with writer.saving(fig, out, dpi=dpi):
        for i in range(1, n):
            text = ax.text(x=0.5, y=0.5, z=-1., s=timesteps[i], size=50, c='black')
            x, y, z = traj_x[i,:,0], traj_x[i,:,1], traj_x[i,:,2]
            c = traj_f[i]
            scatter = ax.scatter(x, y, z, s=50, c=c, cmap=cmap)
            # scatter = ax.scatter(x, y, z, s=50, c=colours)
            u, v, w = traj_grad_f[i,:,0], traj_grad_f[i,:,1], traj_grad_f[i,:,2]
            # quiver = ax.quiver(x, y, z, u, v, w, length=1., normalize=False)
            quiver = ax.quiver(x, y, z, u, v, w, length=0.2, lw=2, normalize=False, cmap=cmap)
            quiver.set_array(c)
            # quiver.set_array(jnp.sqrt(jnp.sum(jnp.square(traj_grad_f[i, :, :]), -1)))
            # break
            writer.grab_frame()
            text.remove()
            scatter.remove()
            quiver.remove()
    # plt.savefig('test.png', dpi=dpi, transparent=False)
    # plt.close(fig)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)

@jax.jit
def vMF_pdf(x, mu, kappa):
    """https://gist.github.com/marmakoide/6f55ff99f14c896399c460a38f72c99a"""
    constant = kappa / ((2 * jnp.pi) * (1. - jnp.exp(-2. * kappa)))
    shape = (x.shape[0], mu.shape[0], x.shape[-1])
    x = jnp.expand_dims(x, 1)
    mu = jnp.expand_dims(mu, 0)
    x = jnp.broadcast_to(x, shape)
    mu = jnp.broadcast_to(mu, shape)
    return constant * jnp.exp(kappa * (batch_mul(mu, x).sum(-1) - 1.))

@jax.jit
def log_prob_vmf(x, mu, kappa):
    output = jnp.log(kappa) - jnp.log(2 * jnp.pi) - kappa - (1 - jnp.exp(- 2 * kappa))
    return output + kappa * batch_mul(mu, x).sum(-1)
