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


def save(ckpt_dir: str, name: str, state) -> None:
    with open(os.path.join(ckpt_dir, f"{name}_arrays.npy"), "wb") as f:
        for x in jax.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, f"{name}_tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def restore(ckpt_dir, name: str):
    with open(os.path.join(ckpt_dir, f"{name}_tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)
 
    leaves, treedef = jax.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, f"{name}_arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_unflatten(treedef, flat_state)