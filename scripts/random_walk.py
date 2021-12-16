from timeit import timeit
import functools
from typing import Tuple

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices

from score_sde.sampling import EulerMaruyamaManifoldPredictor, get_pc_sampler
from score_sde.sde import SDE, batch_mul, Brownian
from score_sde.utils import TrainState, ScoreFunctionWrapper, replicate
from score_sde.models import MLP
import jax
from jax import numpy as jnp
import haiku as hk
import optax
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
        sphere.plot_heatmap(ax, pdf)
    points = gs.to_ndarray(trajectories[0], to_ndim=2)
    sphere.draw(ax, color=color, marker=".")
    scatter = sphere.draw_points(ax, points=points, color=color, marker=".")
    with writer.saving(fig, out, dpi=dpi):
        for points in trajectories[1:]:
            points = gs.to_ndarray(points, to_ndim=2)
            scatter.remove()
            scatter = sphere.draw_points(ax, points=points, color=color, marker=".")
            writer.grab_frame()


def plot_and_save(
    trajectories, pdf=None, size=20, dpi=300, out="out.jpg", color="red"
):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    # colours = sns.cubehelix_palette(n_colors=trajectories.shape[0], light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False)
    colours = ["green", "blue"]
    sphere = visualization.Sphere()
    if pdf:
        sphere.plot_heatmap(ax, pdf)
    sphere.draw(ax, color=color, marker=".")
    for i, points in enumerate(trajectories):
        points = gs.to_ndarray(points, to_ndim=2)
        sphere.draw_points(ax, points=points, color=colours[i], marker=".")
    plt.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def vMF_pdf(x, mu, kappa):
    """https://gist.github.com/marmakoide/6f55ff99f14c896399c460a38f72c99a"""
    constant = kappa / ((2 * np.pi) * (1. - np.exp(-2. * kappa)))
    return constant * np.exp(kappa * (np.dot(mu, x) - 1.))


@jax.jit
def brownian_motion(previous_x, traj, delta, N):
    rng = jax.random.PRNGKey(0)

    def body(step, val):
        rng, x, traj = val
        traj = traj.at[step].set(x)
        # rng, ambiant_noise = gs.random.normal(state=rng, size=(x.shape[0], S2.embedding_space.dim))
        rng, tangent_noise = S2.random_normal_tangent(state=rng, base_point=x, n_samples=x.shape[0])
        x = S2.metric.exp(tangent_vec=delta * tangent_noise, base_point=x)
        return rng, x, traj

    _, x, traj = jax.lax.fori_loop(0, N, body, (rng, previous_x, traj))
    return x, traj


def main():
    """Run gradient descent on a sphere."""
    # gs.random.seed(1985)

    S2 = Hypersphere(dim=2)
    N = 100
    n_samples = 1000
    delta = 0.1
    mu = gs.array([1, 0, 0])
    kappa = 15
    initial_point = S2.random_von_mises_fisher(kappa=kappa, n_samples=n_samples)
    previous_x = initial_point
    traj = jnp.zeros((N + 1, previous_x.shape[0], S2.dim + 1))

    x, traj = brownian_motion(previous_x, traj, delta, N)
    plot_and_save_video(traj, pdf=functools.partial(vMF_pdf, mu=mu, kappa=kappa), out="forward.mp4")


def score_sde():
    manifold = Hypersphere(dim=2)
    N = 1000
    n_samples = 10 ** 3
    # x = manifold.random_von_mises_fisher(kappa=15, n_samples=n_samples)

    sde = Brownian(manifold, T=100, N=N)
    x = sde.prior_sampling(jax.random.PRNGKey(0), (n_samples,))
    # timesteps = jnp.linspace(sde.T, 1e-3, sde.N)
    # predictor = EulerMaruyamaManifoldPredictor(sde, score_fn=None)

    # @jax.jit
    # def loop_body(i, val):
    #     rng, x, x_mean = val
    #     t = timesteps[i]
    #     vec_t = jnp.ones(x.shape[0]) * t
    #     rng, step_rng = random.split(rng)
    #     x, x_mean = predictor.update_fn(step_rng, x, vec_t)
    #     return rng, x, x_mean

    # _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))

    # Score networks
    # score_model = lambda x, t: gs.concatenate([-x[..., [1]], x[..., [0]], gs.zeros((*x.shape[:-1], 1))], axis=-1)  # one of the divergence free vector field, i.e. generate an isometry
    score_model = lambda x, t: manifold.to_tangent(gs.concatenate([gs.zeros((*x.shape[:-1], 1)), 10 * gs.ones((*x.shape[:-1], 1)), gs.zeros((*x.shape[:-1], 1))], axis=-1), x)
    score_model = lambda x, t: manifold.to_tangent(ScoreFunctionWrapper(MLP(hidden_shapes=1*[64], output_shape=3, act='sin'))(x, t), x)
    def score_model(x, t):
        invariant_basis = manifold.invariant_basis(x)
        weights = ScoreFunctionWrapper(MLP(hidden_shapes=1*[64], output_shape=3, act='sin'))(x, t)  # output_shape = dim(Isom(manifold))
        return (gs.expand_dims(weights, -2) * invariant_basis).sum(-1)

    model = hk.transform_with_state(score_model)
    params, state = model.init(rng=jax.random.PRNGKey(0), x=x, t=0)
    # print(model.apply(params, state, jax.random.PRNGKey(0), x=x, t=0)[0].shape)

    optimiser = optax.adam(1e-3)
    opt_state = optimiser.init(params)
    train_state = TrainState(
        opt_state=opt_state, model_state=state, step=0, params=params, ema_rate=0.999, params_ema=params, rng=jax.random.PRNGKey(0)
    )
    p_train_state = replicate(train_state)
    
    sampler = get_pc_sampler(sde, model, (n_samples,), predictor=EulerMaruyamaManifoldPredictor, corrector=None, inverse_scaler=lambda x: x, snr=0.2, continuous=True)
    samples, _ = sampler(replicate(jax.random.PRNGKey(0)), p_train_state)
    print(samples.shape)
    prior_likelihood = lambda x: jnp.exp(sde.prior_logp(x))
    traj = jnp.concatenate([jnp.expand_dims(x, 0), samples], axis=0)
    plot_and_save(traj)


if __name__ == "__main__":
    score_sde()