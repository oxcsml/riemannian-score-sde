#%%
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import setGPU
from functools import partial

import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import vmap, jit, random, lax
from jax.lax import fori_loop

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere, gegenbauer_polynomials
from score_sde.utils import batch_mul
from riemannian_score_sde.sde import VPSDE, Brownian
from score_sde.models.flow import PushForward, CNF
from riemannian_score_sde.sampling import get_pc_sampler

# cmap_name = "plasma_r"
cmap_name = "viridis_r"
#%%
rng = random.PRNGKey(1)
B = 512 * 4
manifold = Hypersphere(2)
# sde = Brownian(manifold, tf=4, t0=0.0, beta_0=1, beta_f=1)
sde = Brownian(manifold, tf=1, t0=0, beta_0=0.001, beta_f=10)
timesteps = jnp.linspace(sde.t0, sde.tf, 20)
timesteps = jnp.expand_dims(timesteps, -1)
x0 = jnp.array([[1.0, 0.0, 0.0]])
x0b = jnp.repeat(x0, B, 0)

# def MMD
# def k(x, y, kappa=0.5):
# return jnp.exp(-jnp.linalg.norm(x - y, axis=-1) ** 2 / kappa**2 / 2)


def k(x, x0, kappa=0.5, n_max=10):
    d = manifold.dim
    n = jnp.expand_dims(jnp.arange(0, n_max + 1), axis=-1)
    # t = jnp.expand_dims(t, axis=0)
    t = jnp.array(kappa**2 / 2)
    coeffs = (
        jnp.exp(-n * (n + 1) * t) * (2 * n + d - 1) / (d - 1) / manifold.metric.volume
    )
    inner_prod = jnp.sum(x0 * x, axis=-1)
    cos_theta = jnp.clip(inner_prod, -1.0, 1.0)
    P_n = gegenbauer_polynomials(alpha=(d - 1) / 2, l_max=n_max, x=cos_theta)
    prob = jnp.sum(coeffs * P_n, axis=0)
    return prob


def mmd(xs, ys):
    @jit
    def k_matrix(xs, ys):
        return vmap(
            lambda x: vmap(lambda y: k(x, y), in_axes=1, out_axes=1)(xs),
            in_axes=1,
            out_axes=1,
        )(ys)

    m = xs.shape[1]
    n = ys.shape[1]

    # biased but positive
    k_xx = k_matrix(xs, xs).sum(axis=(-2, -1))
    k_yy = k_matrix(ys, ys).sum(axis=(-2, -1))
    k_xy = k_matrix(xs, ys).sum(axis=(-2, -1))
    sq_mmd = k_xx / m / m + k_yy / n / n - 2 * k_xy / m / n

    # unbiased (but can be negative)
    # k_xx = k_matrix(xs, xs)
    # k_xx = jnp.sum(k_xx, axis=(-2, -1)) - jnp.trace(k_xx, axis1=-2, axis2=-1)
    # k_yy = k_matrix(ys, ys)
    # k_yy = jnp.sum(k_yy, axis=(-2, -1)) - jnp.trace(k_yy, axis1=-2, axis2=-1)
    # k_xy = k_matrix(xs, ys).sum(axis=(-2, -1))
    # sq_mmd = k_xx / m / (m - 1) + k_yy / n / (n - 1) - 2 * k_xy / m / n

    # return sq_mmd
    return jnp.sqrt(sq_mmd)


# Get "exact" samples (with N = 1000)
N = 1000
rng, next_rng = random.split(rng)
sampler = get_pc_sampler(sde, N, predictor="GRW")
xt_true = vmap(lambda t: sampler(next_rng, x0b, tf=t))(timesteps)
assert vmap(partial(manifold.belongs, atol=1e-5))(xt_true).all()

# Get approximate samples (sweeping over N)
Ns = np.array([1, 2, 5, 50, 100, 1000])
# Ns = np.array([2, 5, 10, 20, 50, 100])

fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharey=True, sharex=True)
colors = sns.color_palette(cmap_name, len(Ns))
fontsize = 30

for i, N in enumerate(Ns):
    xt_approx = vmap(
        lambda t: get_pc_sampler(sde, N, predictor="GRW")(
            next_rng, x0b, tf=t
        )
    )(timesteps)
    assert vmap(partial(manifold.belongs, atol=1e-5))(xt_approx).all()
    dists = mmd(xt_approx, xt_true)

    ax.plot(timesteps, dists, color=colors[i], label=f"N={N}", lw=5)

ax.set_xlabel("t", fontsize=fontsize)
ax.set_ylabel(
    r"MMD$(\hat{\mathbf{X}}_t|\mathbf{X}_0, \mathbf{X}_t|\mathbf{X}_0)$",
    fontsize=fontsize,
)
ax.set_yscale("log")
ax.legend(fontsize=4 / 5 * fontsize, loc="upper right")
ax.tick_params(axis="both", which="major", labelsize=4 / 5 * fontsize)

# fig.tight_layout(pad=0.5)
fig_name = f"../doc/images/approximate_forward.pdf"
fig.savefig(fig_name, bbox_inches="tight", transparent=True)
