# %%
%load_ext autoreload
%autoreload 2

from functools import partial
import os
os.environ["GEOMSTATS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp

from geomstats.geometry.hyperbolic import Hyperbolic, PoincareBall, Hyperboloid
from geomstats.geometry._hyperbolic import _Hyperbolic
from riemannian_score_sde.models.distribution import WrapNormDistribution

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

#%%

to_ball = _Hyperbolic._extrinsic_to_ball_coordinates
to_extr = _Hyperbolic._ball_to_extrinsic_coordinates

def proj(h, x):
    return to_ball(x) if isinstance(h, Hyperboloid) else x

def lift(h, x):
    return to_extr(x) if isinstance(h, Hyperboloid) else x

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


#%%
%matplotlib inline

#%%
# distribution of radius

rng = jax.random.PRNGKey(42)
_, axes = plt.subplots(1, 2, figsize=(25, 10), sharex=True, sharey=True)
scale = 0.5

for i, coords in enumerate(['ball', 'extrinsic']):
    h = Hyperbolic(dim=2, default_coords_type=coords)
    mean = lift(h, jnp.array([0.5, 0.0]))
    dist = WrapNormDistribution(h, scale=scale, mean=mean)
    samples = dist.sample(rng, 10000)

    d = h.metric.dist(samples, mean[None,...])
    x = jnp.linspace(0,jnp.max(d),1000)
    y = x / scale ** 2 * jnp.exp(-x **2 / 2 / scale ** 2)
    # vs = x[...,None] * lift(h, jnp.array([1.0, 0.0]))[None,...]
    # xs = h.metric.exp(vs, h.identity)
    # y = jnp.exp(dist.log_prob(xs))
    axes[i].hist(d, density=True, bins=100)
    axes[i].plot(x, y, color='black', lw=5)
    print(f"var: {scale**2:.2f} vs {((d**2).mean()/2).item():.2f}")


#%%
h = Hyperbolic(dim=2, default_coords_type='ball')
# h = Hyperbolic(dim=2, default_coords_type='extrinsic')
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
from score_sde.schedule import LinearBetaSchedule
from riemannian_score_sde.sde import Langevin

beta_schedule = LinearBetaSchedule(beta_0=0.01, beta_f=5.0, tf=1)
sde = Langevin(beta_schedule, h, ref_scale=0.5, ref_mean=lift(h, jnp.array([0.0, 0.0])), N=10000)

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
dist = WrapNormDistribution(h, scale=0.5, mean=lift(h, jnp.array([0.8, 0.0])))
_, tracks, _ = sde.marginal_sample(jax.random.PRNGKey(0), dist.sample(rng, 10000), t=1, return_hist=True)
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

#%%
# distribution of radius

rng = jax.random.PRNGKey(42)

from score_sde.schedule import LinearBetaSchedule
from riemannian_score_sde.sde import Langevin

beta_schedule = LinearBetaSchedule(beta_0=0.01, beta_f=5.0, tf=1)
sde = Langevin(beta_schedule, h, ref_scale=.5, ref_mean=lift(h, jnp.array([0.0, 0.0])), N=1000)

x1 = sde.marginal_sample(rng, lift(h, jnp.zeros((10000, 2))), t=1.0, return_hist=False)
x2 = sde.sample_limiting_distribution(rng, 10000)
d1 = h.metric.dist(x1, sde.limiting.mean[None,...])
d2 = h.metric.dist(x2, sde.limiting.mean[None,...])
# d1 = h.metric.dist(lift(h,tracks[-1, ...]), sde.limiting.mean[None,...])
# d2 = h.metric.dist(lift(h,samples), sde.limiting.mean[None,...])
_, axes = plt.subplots(1, 2, figsize=(25, 10), sharex=True, sharey=True)
x = jnp.linspace(0,jnp.max(d2),1000)
scale = sde.limiting.scale.mean()
y = x / scale ** 2 * jnp.exp(-x **2 / 2 / scale ** 2)
for ax, d in zip(axes, [d1, d2]):
    ax.hist(d, density=True, bins=100)
    ax.plot(x, y, color='black', lw=5)


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
_, axes = plt.subplots(1, 5, figsize=(25, 5+1))
for ax, i in zip(axes, [0,2000,4000,6000,9990]):
    ax.scatter(tracks[i, :, 0], tracks[i, :, 1], alpha=0.3, s=2)
    make_circle(ax)

#%%
## debug when computing jacbobian of forward drift

import geomstats.algebra_utils as utils

# fwd_drift = lambda x: sde.coefficients(x, 0.5 * jnp.ones((x.shape[0])))

def dist(point_a, point_b):
    sq_norm_a = sde.manifold.metric.embedding_metric.squared_norm(point_a)
    sq_norm_b = sde.manifold.metric.embedding_metric.squared_norm(point_b)
    inner_prod = sde.manifold.metric.embedding_metric.inner_product(point_a, point_b)

    cosh_angle = -inner_prod / jnp.sqrt(sq_norm_a * sq_norm_b)
    cosh_angle = jnp.clip(cosh_angle, 1.0 + 1e-7, 1e24)
    # return cosh_angle

    dist = jnp.arccosh(cosh_angle)
    dist *= sde.manifold.metric.scale
    return dist

def logdetexp(x):
    # d = h.metric.dist(x, sde.limiting.mean)
    d = dist(x, sde.limiting.mean)
    # log_sinch = jnp.log(jnp.sinh(d) / d)
    log_sinch = utils.taylor_exp_even_func(d**2, utils.log_sinch_close_0)
    # log_sinch = taylor_exp_even_func(d**2, log_sinch_close_0)
    # log_sinch = 1/6 * d**2# - 1/180 * d**4 + 1/2835 * d**6 - 1/37800 * d**8
    return (2 - 1) * log_sinch


# logdetexp = lambda x: sde.manifold.metric.logdetexp(sde.limiting.mean, x)
# sq_dist = lambda x:  sde.manifold.metric.dist(x, sde.limiting.mean) ** 2
sq_dist = lambda x:  dist(x, sde.limiting.mean) ** 2

def logU(x):
    # NOTE: scale must be isotropic!
    res = 0.5 * sq_dist(x) / (sde.limiting.scale.mean() ** 2)
    # return logdetexp(x)
    return res + logdetexp(x)
