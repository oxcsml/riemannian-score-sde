#%%
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
import setGPU
from functools import partial

import math
import numpy as np
import jax
from jax import vmap, jit, random, lax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.special import erf

from matplotlib import pyplot as plt
import seaborn as sns

from geomstats.geometry.euclidean import Euclidean
from score_sde.utils import batch_mul
from riemannian_score_sde.sde import VPSDE
from riemannian_score_sde.sampling import get_pc_sampler

# cmap_name = "RdBu"
cmap_name = "plasma_r"
cmap_name = "viridis_r"

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = ["Computer Modern Roman"]
# plt.rcParams.update(
#     {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
# )

#%%
class Normal:
    def __init__(self, mean=0, scale=1):
        self.mean = mean
        self.scale = scale

    def prob(self, x):
        return norm.pdf(x, loc=self.mean, scale=self.scale)

    def sample(self, rng, shape):
        z = random.normal(rng, shape)
        return self.mean + z * self.scale


p_0 = Normal(8, 1)
p_ref = Normal(0, 1)


def p_moser(x, t):
    return (1 - t) * p_0.prob(x) + t * p_ref.prob(x)


def p_sgm(x, t, a):
    # X_t = e^-t X_0 + (1 - e^{-2t})^{1/2} Z ~ N(a * e^{-t}, 1)
    # return norm.pdf(x, loc=a * jnp.exp(-t), scale=1)
    return norm.pdf(x, loc=a * (1 - t), scale=1)


ts = jnp.linspace(0, 1, 11, endpoint=True)
xs = jnp.linspace(-4, 12, 100)

ps_moser = vmap(lambda t: vmap(lambda x: p_moser(x, t))(xs))(ts)
ps_sgm = vmap(lambda t: vmap(lambda x: p_sgm(x, t, p_0.mean))(xs))(ts)

### Plotting
p_dict = {"moser": ps_moser, "sgm": ps_sgm}
name_dict = {"moser": "Moser flow", "sgm": "Score-based generative model"}

fig, axis = plt.subplots(1, 2, figsize=(15, 5), sharey=True, sharex=True)
colors = sns.color_palette(cmap_name, len(ts))
fontsize = 20

for i, (ax, method) in enumerate(zip(axis, ["moser", "sgm"])):
    for k, t in enumerate(ts):
        ax.plot(xs, p_dict[method][k], c=colors[k], label=f"t={t:.1f}", lw=3, alpha=0.5)

    # ax.set_xlabel("x", fontsize=fontsize)
    if i == 0:
        # ax.legend(fontsize=0.8 * fontsize)
        ax.set_ylabel("Probability density", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=4 / 5 * fontsize)
    # ax.tick_params(axis="both", which="minor", labelsize=0.8 * 15)
    ax.set_title(f"{name_dict[method]}", fontsize=fontsize)
    ax.set_xlabel(r"$x$", fontsize=fontsize)
    ax.set_xlim([-4, 12])
    ax.set_xticks([-4, 0, 4, 8, 12], [-4, 0, 4, 8, 12])
# plt.legend(
#     bbox_to_anchor=(1.0, 1.0),
#     loc="upper left",
#     fontsize=0.8 * fontsize,
# )

fig.tight_layout(pad=1)
fig_name = f"../doc/images/horizontal_vertical_pdf.pdf"
fig.savefig(fig_name, bbox_inches="tight", transparent=True)


def sgm_score(sde, a, x, t):
    # X_t = e^-t X_0 + (1 - e^{-2t})^{1/2} ~ N(a * e^{-t}, 1)
    t = jnp.array(t)
    t = sde.rescale_t(t)
    mean = a * jnp.exp(-t)
    std = jnp.ones_like(mean)
    score = -1 / (std**2) * (x - mean)
    return score


def get_scores_sgm(rng, a):
    rng, next_rng = random.split(rng)
    x0 = Normal(a, 1).sample(next_rng, (B, d))

    sampler = get_pc_sampler(sde, N, predictor="GRW", return_hist=True)
    rng, next_rng = random.split(rng)
    _, xt, timesteps = sampler(next_rng, x0, tf=sde.tf)

    xt = xt.reshape((-1, 1))
    timesteps = timesteps.reshape((-1, 1))
    logp_grad = vmap(partial(sgm_score, sde, a))(xt, timesteps)

    sq_norm = manifold.metric.squared_norm(logp_grad, xt)
    sq_norm = sq_norm.reshape((N, B)).mean(-1)
    ts = timesteps.reshape((N, B))[..., 0]
    xt = xt.reshape((N, B))
    return xt, sq_norm, ts


def moser_score(a, x: jnp.ndarray, t: float) -> jnp.ndarray:
    t = t.reshape((-1, 1))
    # alpha_t = t * nu + (1 - t) * mu_plus
    # out = -u / alpha_t
    p_0 = Normal(a, 1)
    alpha_t = (1 - t) * p_0.prob(x) + t * p_ref.prob(x)
    u = 1 / (2 * math.sqrt(2)) * (erf(x - a) - erf(x))
    out = u / alpha_t
    return out


def get_scores_moser(a, rng, t):
    means = jnp.array([0.0, a])
    stds = jnp.array([1.0, 1.0])
    weights = jnp.array([t, 1 - t]).reshape(-1)
    rng, next_rng = random.split(rng)
    indices = jax.random.choice(next_rng, a=len(weights), shape=(B,), p=weights)
    rng, next_rng = random.split(rng)
    xt = means[indices] + stds[indices] * jax.random.normal(
        next_rng, shape=(B,) + means.shape[1:]
    )
    score_t = moser_score(a, xt, t).reshape(B, -1)
    sq_norm_t = manifold.metric.squared_norm(score_t, xt).mean(axis=(-1))
    # sq_norm = sq_norm.at[i].set(sq_norm_t)
    return xt, sq_norm_t, t


rng = random.PRNGKey(0)

B = 512 * 16
N = 1000
d = 1
manifold = Euclidean(dim=d)
# sde = VPSDE(manifold=manifold, tf=5, beta_0=1, beta_f=1)
sde = VPSDE(manifold=manifold, tf=1, beta_0=1e-3, beta_f=12)

eps = 1e-3
tf = 1
dt = (tf - eps) / N
timesteps = jnp.expand_dims(jnp.linspace(eps, tf, N), -1)
rng, next_rng = random.split(rng)

fig, axis = plt.subplots(1, 2, figsize=(15, 5), sharey="row", sharex=False)
# p0_means = jnp.linspace(1, 10, 8, endpoint=True)
p0_means = [1.0, 2.0, 5.0, 8.0]

for i, (ax, model) in enumerate(zip(axis, p_dict.keys())):
    for k, a in enumerate(p0_means):
        if model == "sgm":
            xt, sq_norm, ts = get_scores_sgm(rng, a)
        else:
            xt, sq_norm, ts = vmap(partial(get_scores_moser, a))(
                random.split(rng, num=N), timesteps
            )
        sq_norm = jnp.sqrt(sq_norm)
        label = r"$\mathbb{E}[p_0]$" + f"={a:}"
        colors = sns.color_palette(cmap_name, len(p0_means))
        ax.plot(ts[1:-1], sq_norm[1:-1], c=colors[k], label=label, lw=3)
    ax.set_yscale("log")
    if i == 0:
        ax.set_ylabel(
            r"$\mathbb{E}\left[\|\|\nabla \log p_t(\mathbf{X}_t)\|\|\right]$",
            fontsize=fontsize,
            rotation=90,
            # labelpad=2 / 3 * fontsize,
        )
    else:
        ax.legend(fontsize=0.8 * fontsize)
    ax.set_xlabel(r"$t$", fontsize=fontsize)

    ax.tick_params(axis="both", which="major", labelsize=4 / 5 * fontsize)
    ax.set_title(f"{name_dict[model]}", fontsize=fontsize)

fig.tight_layout(pad=0.5)
fig_name = f"../doc/images/horizontal_vertical_norm.pdf"
fig.savefig(fig_name, bbox_inches="tight", transparent=True)


fig, axis = plt.subplots(1, 2, figsize=(15, 5), sharey="row", sharex=False)

a = 8
for i, (ax, model) in enumerate(zip(axis, p_dict.keys())):
    if model == "sgm":
        xt, sq_norm, ts = get_scores_sgm(rng, a)
    else:
        xt, sq_norm, ts = vmap(partial(get_scores_moser, a))(
            random.split(rng, num=N), timesteps
        )
    sq_norm = jnp.sqrt(sq_norm)

    indices = jnp.arange(N)[(jnp.arange(N) % 100) == 0]
    indices = jnp.concatenate([indices, jnp.array([N])])
    colors = sns.color_palette(cmap_name, len(indices))
    for k, idx in enumerate(indices):
        ax.hist(
            np.array(xt[idx]),
            bins=round(math.sqrt(B) * 0.8),
            density=True,
            color=colors[k],
            alpha=0.2,
            label=f"t={ts[idx].item():.1f}",
        )
    if i == 0:
        ax.set_ylabel(
            r"$X_t$",
            fontsize=fontsize,
            rotation=0,
            labelpad=2 / 3 * fontsize,
        )
    # else:
    #     ax.legend(fontsize=0.8 * fontsize)

    ax.set_xlabel(r"$x$", fontsize=fontsize)
    ax.set_xlim([-4, 12])
    ax.set_xticks([-4, 0, 4, 8, 12], [-4, 0, 4, 8, 12])

    ax.tick_params(axis="both", which="major", labelsize=4 / 5 * fontsize)
    ax.set_title(f"{name_dict[model]}", fontsize=fontsize)

fig.tight_layout(pad=0.5)
fig_name = f"../doc/images/horizontal_vertical_hist.pdf"
fig.savefig(fig_name, bbox_inches="tight", transparent=True)

# %%
