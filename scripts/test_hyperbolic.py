# %%
%load_ext autoreload
%autoreload 2

# %%
from functools import partial
import os
os.environ["GEOMSTATS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import jax
import jax.numpy as jnp

from geomstats.geometry.hyperbolic import Hyperbolic, PoincareBall, Hyperboloid
from geomstats.geometry._hyperbolic import _Hyperbolic
from riemannian_score_sde.models.distribution import WrapNormDistribution

to_ball = _Hyperbolic._extrinsic_to_ball_coordinates
to_extr = _Hyperbolic._ball_to_extrinsic_coordinates

def proj(h, x):
    return to_ball(x) if isinstance(h, Hyperboloid) else x

def lift(h, x):
    return to_extr(x) if isinstance(h, Hyperboloid) else x

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

rng = jax.random.PRNGKey(0)

def make_circle(ax=None):
    theta = jnp.linspace(0, 2*jnp.pi, 100)
    if ax is not None:
        # ax.plot(jnp.sin(theta), jnp.cos(theta), color='black')
        ax.add_patch(Circle((0, 0), 1., color="black", fill=False, linewidth=2, zorder=10))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
    else:
        plt.plot(jnp.sin(theta), jnp.cos(theta), color='black')
        # plt.add_patch(Circle((0, 0), 1., color="black", fill=False, linewidth=2, zorder=10))
        plt.gca().set_aspect('equal')
        plt.axis('off')

from riemannian_score_sde.utils.vis import make_disk_grid

def disk_plot(ax, h, prob_fn, N=150):
    cmap = sns.cubehelix_palette(light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True)
    make_circle(ax)

    xs, volume, lam = make_disk_grid(N)

    prob = jnp.exp(prob_fn(lift(h, xs)))# * mask 
    prob = prob.at[jnp.isnan(prob)].set(0.0)
    measure = prob * lam# ** 2 * mask
    Z = (measure * volume).mean()
    # Z = measure.mean() #TODO: should it be multiplied by the volume?
    print("volume = {:.2f}".format(Z.item()))

    xs = xs.reshape(N, N, 2)
    measure = measure.reshape(N, N)
    ax.pcolormesh(xs[:,  :, 0], xs[:,  :, 1], measure, cmap=cmap, linewidth=0, rasterized=True, shading="gouraud")


# %%
%matplotlib inline
#%%
# h = Hyperbolic(dim=2, default_coords_type='ball')
h = Hyperbolic(dim=2, default_coords_type='extrinsic')
is_hyperboloid = isinstance(h, Hyperboloid)

means = [0.0, 0.0, 0.5, 0.5, 0.8, 0.8]
scales = [0.2, 0.5, 0.2, 0.5, 0.2, 0.5]

fig, axes = plt.subplots(2, 6, figsize=(30, 10))
for ax, mean_x, scale in zip(axes[0], means, scales):
    mean = lift(h, jnp.array([mean_x, 0.0]))
    dist = WrapNormDistribution(h, scale=scale, mean=mean)
    samples = proj(h, dist.sample(rng, 10000))
    ax.scatter(samples[..., 0], samples[..., 1], alpha=0.3, s=2)
    make_circle(ax)

for ax, mean_x, scale in zip(axes[1], means, scales):
    mean = jnp.array([mean_x, 0.0])
    mean = to_extr(mean) if is_hyperboloid else mean
    dist = WrapNormDistribution(h, scale=scale, mean=mean)
    disk_plot(ax, h, dist.log_prob)

# %% 
# Limiting distribution density and samples

from score_sde.sde import SDE
from score_sde.schedule import LinearBetaSchedule, QuadraticSchedule
from riemannian_score_sde.sde import NonCompactWrapNorm

# beta_schedule = LinearBetaSchedule(beta_0=0.01, beta_f=5.0, tf=1)
beta_schedule = QuadraticSchedule(tf=1, beta_0=0.01, beta_f=5.0)
sde = NonCompactWrapNorm(beta_schedule, h, ref_scale=0.5, ref_mean=lift(h, jnp.array([0.0, 0.0])), N=10000)

_, axes = plt.subplots(1, 2, figsize=(12, 6))
samples = proj(h, sde.sample_limiting_distribution(rng, 10000))
axes[0].scatter(samples[..., 0], samples[..., 1], alpha=0.3, s=2)
make_circle(axes[0])
disk_plot(axes[1], h, sde.limiting_distribution_logp)

# %% 
# Trajectories x_t|x_0
points = sde.marginal_sample(jax.random.PRNGKey(0), lift(h, jnp.zeros((1000, 2))), t=1.0, return_hist=False)
points = proj(h, points)

plt.scatter(points[..., 0], points[..., 1], alpha=0.5, s=2)
make_circle()

M = 3
start = lift(h, jnp.repeat(jnp.array([0.8, 0.0])[None, ...], M, 0))
_, tracks, _ = sde.marginal_sample(jax.random.PRNGKey(0), start, t=2, return_hist=True)
tracks = proj(h, tracks)
_, axes = plt.subplots(1, 3, figsize=(12, 6))
for k, ax in enumerate(axes):
    max_k = int(len(tracks) * (k+1) / 3)
    for i in range(M):
        ax.plot(tracks[:max_k, i, 0], tracks[:max_k, i, 1], alpha=0.5, linewidth=1)
    make_circle(ax)

# %%
# Test that limiting distribution is the invariant/limiting distribution

# Converges to ref dist from a different dist
# dist = WrapNormDistribution(h, scale=0.5, mean=lift(h, jnp.array([0.8, 0.0])))
# samples = dist.sample(jax.random.PRNGKey(0), 10000)

from riemannian_score_sde.datasets.simple import WrapNormMixtureDistribution
mean = lift(h, 0.4*jnp.array([[-1, 0.0],[1, 0.0],[0.0, -1],[0.0, 1]]))
scale = jnp.array([[0., 0.15, 0.5],[0., 0.15, 0.5],[0., 0.5, 0.15],[0., 0.5, 0.15]])/2
dist = WrapNormMixtureDistribution(10000, h, mean, scale)
samples = next(dist)[0]

beta_schedule = QuadraticSchedule(tf=1, beta_0=0.01, beta_f=1.0)
sde = NonCompactWrapNorm(beta_schedule, h, ref_scale=0.5, ref_mean=lift(h, jnp.array([0.0, 0.0])), N=10000)

_, tracks, _ = sde.marginal_sample(jax.random.PRNGKey(0), samples, t=1, return_hist=True)
tracks = proj(h, tracks)
samples = proj(h, sde.sample_limiting_distribution(rng, 10000))

_, axes = plt.subplots(1, 5+1, figsize=(25, 5+1))
for ax, i in zip(axes[:-1], [0,2000,4000,6000,9990]):
    ax.scatter(tracks[i, :, 0], tracks[i, :, 1], alpha=0.3, s=2)
    make_circle(ax)
axes[-1].scatter(samples[..., 0], samples[..., 1], alpha=0.3, s=2, color='orange')
make_circle(axes[-1])

# Ref dist + P_0 dist match -> stays constant

samples = sde.sample_limiting_distribution(rng, 10000)
_, tracks, _ = sde.marginal_sample(jax.random.PRNGKey(0), samples, t=1, return_hist=True)
tracks = proj(h, tracks)
samples = proj(h, samples)

_, axes = plt.subplots(1, 5+1, figsize=(25, 5+1))
for ax, i in zip(axes, [0,2000,4000,6000,9990]):
    ax.scatter(tracks[i, :, 0], tracks[i, :, 1], alpha=0.3, s=2)
    make_circle(ax)
axes[-1].scatter(samples[..., 0], samples[..., 1], alpha=0.3, s=2, color='orange')
make_circle(axes[-1])

# %%
# Reverse process with score network
from riemannian_score_sde.sampling import get_pc_sampler

def score_fn(x, t):
    # zero = jnp.zeros(h.dim)
    # res = -self.manifold.metric.log(x, zero)
    fwd_drift, diffusion = sde.coefficients(x, t)
    tangent = h.metric.log(dist.mean, x) / (dist.scale**2)#/ (diffusion[...,None] ** 2)
    # tangent = self.net(x, t)
    # tangent = jnp.ones_like(x) * 1
    # tangent = h.metric.transpfrom0(x, tangent)
    # return tangent
    return fwd_drift / (diffusion[...,None] ** 2) + tangent
    # return fwd_drift / (diffusion[...,None] ** 2) + tangent# / (diffusion[...,None] ** 2)

rsde = sde.reverse(score_fn)
sampler = get_pc_sampler(
    rsde,
    sde.N,
    predictor="GRW",
    return_hist=True
)
rng, next_rng = jax.random.split(rng)
x0, tracks, _ = sampler(next_rng, lift(h, samples), t0=2)

norm = jnp.linalg.norm(x0, axis=-1)
prop_close_to_border = (jnp.abs(norm - 1.) < 1e-5).mean() * 100
print("prop_close_to_border", prop_close_to_border)

tracks = proj(h, tracks)
print(tracks.shape)
M=5
_, axes = plt.subplots(1, M, figsize=(25, 5+1))
for i, ax in enumerate(axes):
    i = int(i/(M-1) * tracks.shape[0])
    ax.scatter(tracks[i, :, 0], tracks[i, :, 1], alpha=0.3, s=2)
    make_circle(ax)

# %% load model
import os
import socket
import logging
from timeit import default_timer as timer
from tqdm import tqdm

import jax
from jax import numpy as jnp
import optax
import haiku as hk

from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class, call

from score_sde.models.flow import SDEPushForward
from score_sde.losses import get_ema_loss_step_fn
from score_sde.utils import TrainState, save, restore
from score_sde.utils.loggers_pl import LoggerCollection
from score_sde.datasets import random_split, DataLoader, TensorDataset
from riemannian_score_sde.utils.normalization import compute_normalization
from riemannian_score_sde.utils.vis import plot, plot_ref

run_path = "/data/ziz/not-backed-up/mhutchin/score-sde/results/hyp/beta_schedule.beta_f=2,flow.ref_scale=0.5,steps=100000/0"

cfg = OmegaConf.load(run_path + "/.hydra/config.yaml")
ckpt_path = os.path.join(run_path, cfg.ckpt_dir)
os.makedirs(ckpt_path, exist_ok=True)

rng = jax.random.PRNGKey(cfg.seed)
data_manifold = instantiate(cfg.manifold)
transform = instantiate(cfg.transform, data_manifold)
model_manifold = transform.domain
beta_schedule = instantiate(cfg.beta_schedule)
flow = instantiate(cfg.flow, manifold=model_manifold, beta_schedule=beta_schedule)
base = instantiate(cfg.base, model_manifold, flow)
pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

rng, next_rng = jax.random.split(rng)
dataset = instantiate(cfg.dataset, rng=next_rng)

if isinstance(dataset, TensorDataset):
    # split and wrapp dataset into dataloaders
    train_ds, eval_ds, test_ds = random_split(
        dataset, lengths=cfg.splits, rng=next_rng
    )
    train_ds, eval_ds, test_ds = (
        DataLoader(train_ds, batch_dims=cfg.batch_size, rng=next_rng, shuffle=True),
        DataLoader(eval_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
        DataLoader(test_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
    )
else:
    train_ds, eval_ds, test_ds = dataset, dataset, dataset


def model(y, t, context=None):
    """Vector field s_\theta: y, t, context -> T_y M"""
    output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
    score = instantiate(
        cfg.generator,
        cfg.architecture,
        cfg.embedding,
        output_shape,
        manifold=model_manifold,
    )
    # TODO: parse context into embedding map
    if context is not None:
        t_expanded = jnp.expand_dims(t.reshape(-1), -1)
        if context.shape[0] != y.shape[0]:
            context = jnp.repeat(jnp.expand_dims(context, 0), y.shape[0], 0)
        context = jnp.concatenate([t_expanded, context], axis=-1)
    else:
        context = t
    return score(y, context)

model = hk.transform_with_state(model)

rng, next_rng = jax.random.split(rng)
t = jnp.zeros((cfg.batch_size, 1))
data, context = next(train_ds)
params, state = model.init(rng=next_rng, y=transform.inv(data), t=t, context=context)

schedule_fn = instantiate(cfg.scheduler)
optimiser = optax.chain(instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn))
opt_state = optimiser.init(params)

train_state = restore(ckpt_path)

# %%
from score_sde.models import get_score_fn
from score_sde.sampling import get_pc_sampler
from functools import partial

score_fn = get_score_fn(flow, model, train_state.params_ema, train_state.model_state)
score_fn = partial(score_fn, context=None)

sde = flow
rsde = sde.reverse(score_fn)
sampler = get_pc_sampler(
    rsde,
    sde.N,
    predictor="GRW",
    return_hist=True
)
rng, next_rng = jax.random.split(rng)
x0, tracks, _ = sampler(next_rng, sde.limiting.sample(rng, 1000), t0=2)

M=5
tracks_ = proj(data_manifold, tracks)
print(tracks.shape)
_, axes = plt.subplots(1, M, figsize=(25, 5+1))
for i, ax in enumerate(axes):
    i = int(i/(M-1) * tracks_.shape[0])
    ax.scatter(tracks_[i, :, 0], tracks_[i, :, 1], alpha=0.3, s=2)
    make_circle(ax)

plt.show()
# %%
t0 = sde.tf
tf = sde.t0
t0 = jnp.broadcast_to(t0, tracks.shape[1])
tf = jnp.broadcast_to(tf, tracks.shape[1])
tf = tf + 1e-3
timesteps = jnp.linspace(start=t0, stop=tf, num=sde.N, endpoint=True)

scores = []
for i in range(tracks.shape[0]):
    scores.append(score_fn(tracks[i], timesteps[i]))

scores = jnp.array(scores)
# %%
std = sde.marginal_prob(jnp.zeros_like(tracks), timesteps)[1]

fwd_drift, diffusion = jax.vmap(sde.coefficients)(tracks, timesteps)
residual = 2 * fwd_drift / (diffusion[..., None] ** 2)

raw_scores = (scores - residual) * std[..., None]
# %%
norm_scores = jnp.linalg.norm(raw_scores, axis=-1)
print(jnp.mean(norm_scores), jnp.std(norm_scores), jnp.max(norm_scores), jnp.min(norm_scores))
# %%
plt.plot(norm_scores.mean(axis=1))
# %%
plt.plot(jnp.linalg.norm(scores, axis=-1).mean(axis=1))
# %%
plt.imshow(jnp.linalg.norm(scores, axis=-1).T)
# %%
from riemannian_score_sde.utils.vis import plot_poincare
h = Hyperbolic(dim=2, default_coords_type='ball')
# h = Hyperbolic(dim=2, default_coords_type='extrinsic')
is_hyperboloid = isinstance(h, Hyperboloid)

lim=1.5
points=25
line = jnp.linspace(-lim, lim, points)
tangent_grid = jnp.stack(jnp.meshgrid(line, line), axis=-1)
grid = h.exp(tangent_grid, jnp.zeros_like(tangent_grid))
grid_ = grid.reshape((-1, grid.shape[-1]))

plt.scatter(grid[..., 0], grid[..., 1])
make_circle()

# %%

vec_field = jax.random.normal(jax.random.PRNGKey(0), grid_.shape)

metric_matrix = h.metric.metric_matrix(grid_)

def vec_field(x):
    return jnp.stack(
        [
            x[..., 1],
            -x[..., 0]
        ],
        axis=-1
    )

vf = vec_field(grid_)

plt.quiver(grid_[..., 0], grid_[..., 1], vf[..., 0], vf[..., 1])
make_circle()

def div_fn(f, manifold):
    def euc_div_fn(y: jnp.ndarray, t: float, context: jnp.ndarray):
        y_shape = y.shape
        dim = np.prod(y_shape[1:])
        t = jnp.expand_dims(t.reshape(-1), axis=-1)
        y = jnp.expand_dims(y, 1)  # NOTE: need leading batch dim after vmap
        if context is not None:
            context = jnp.expand_dims(context, 1)
        t = jnp.expand_dims(t, 1)
        jac = jax.vmap(jax.jacrev(fn, argnums=0))(y, t, context)

        jac = jac.reshape([y_shape[0], dim, dim])
        return jnp.trace(jac, axis1=-1, axis2=-2)

    def christoffel_fn(y, f, context):
        

# %%
line = jnp.linspace(1, 10, 1000)

plt.plot(line, jnp.arccosh(line))
# %%
plt.plot(line, jax.vmap(jax.grad(jnp.arccosh))(line))
# %%
jax.vmap(jax.grad(jnp.arccosh))(line)
# %%
line
# %%
jax.grad(jnp.arccosh)(1+1e-8)
# %%
