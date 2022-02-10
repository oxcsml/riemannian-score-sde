import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
from functools import partial
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

from geomstats.geometry.hypersphere import Hypersphere

import jax
from jax import numpy as jnp
import numpy as np
from scipy.stats import gaussian_kde

# from score_sde.sde import SDE
from score_sde.utils import batch_mul
from riemannian_score_sde.sde import Brownian
from scipy.interpolate import splev, splrep

from scripts.utils import (
    plot_and_save2,
    plot_and_save_video,
    plot_and_save_video2,
    vMF_pdf,
)

import matplotlib

# matplotlib.use("pgf")

plt.rcParams["text.usetex"] = True
# plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]
# plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["font.family"] = ["Computer Modern Roman"]
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({"font.size": 10})

# plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]
# # Options
# params = {
#     "text.usetex": True,
#     "font.size": 11,
#     "font.family": "lmodern",
#     # "text.latex.unicode": True,
# }
# plt.rcParams.update(params)

# def plot(x, ys, dim_0_names, dim_1_names, out):
#     fontsize = 12
#     K, J, _ = ys.shape
#     fig, axis = plt.subplots(
#         # nrows=1, ncols=K, figsize=(12, 5), sharex=True, sharey=False
#         nrows=K, ncols=1, figsize=(3, 6), sharex=True, sharey=False
#     )
#     axis = axis if isinstance(axis, np.ndarray) else [axis]
#     # colours = sns.cubehelix_palette(n_colors=J, light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False)
#     colours = sns.cubehelix_palette(n_colors=J, reverse=False)
#     # colours = ["green", "blue"]
#     for k in range(K):
#         for j in range(J):
#             linestyle = "--" if j == J - 1 else "-"
#             axis[k].plot(
#                 x,
#                 ys[k, j],
#                 color=colours[j],
#                 alpha=0.8,
#                 label=dim_1_names[j],
#                 linestyle=linestyle,
#                 lw=3,
#             )
#         axis[k].set_title(dim_0_names[k]) #, y=-0.01)
#     axis[-1].legend(loc="best", fontsize=fontsize*4/5)
#     axis[-1].set_xlabel(r"Signed distance$(x_0, x)$", fontsize=fontsize)
#     plt.xticks([-math.pi/4, 0., math.pi/4], [r"$-\pi/4$", 0, r"$\pi/4$"], fontsize=fontsize)
#     axis[1].set_ylabel("Density", fontsize=fontsize)
#     plt.savefig("{}.png".format(out), dpi=300, bbox_inches="tight")
#     plt.clf()
#     plt.close("all")


def plot(x, ys, dim_0_names, dim_1_names, out):
    K, J, _ = ys.shape
    fig, axis = plt.subplots(
        nrows=1,
        ncols=K,
        figsize=(6.2689, 2),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    axis = axis if isinstance(axis, np.ndarray) else [axis]
    # colours = sns.cubehelix_palette(n_colors=J, light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False)
    colours = sns.cubehelix_palette(n_colors=J, reverse=False)
    # colours = ["green", "blue"]

    for k in range(K):
        for j in range(J):
            linestyle = "--" if j == J - 1 else "-"
            axis[k].plot(
                x,
                ys[k, j],
                color=colours[j],
                alpha=0.8,
                label=dim_1_names[j],
                linestyle=linestyle,
                lw=3,
            )
        axis[k].set_title(dim_0_names[k], fontsize=11)  # , y=-0.01)
        axis[k].tick_params(axis="both", which="major")
    axis[-1].legend(loc="best", fontsize=8)
    axis[1].set_xlabel(r"Signed $d_{\mathcal{M}}(x_0, x_t)$")
    plt.xticks([-math.pi / 4, 0.0, math.pi / 4], [r"$-\pi/4$", 0, r"$\pi/4$"])
    axis[0].set_ylabel("Density")
    # plt.tight_layout()
    # plt.subplots_adjust(
    #     left=None, bottom=None, right=None, top=None, wspace=None, hspace=None
    # )
    plt.savefig("{}.pdf".format(out), dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close("all")


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


# @partial(jax.jit, static_argnums=(3,4,5))
# def heat_kernel(y, s, x, thresh, n_max, manifold):
#     return jnp.exp(manifold.log_heat_kernel(x, y, s, thresh=thresh, n_max=n_max))


@partial(jax.jit, static_argnums=(3, 4, 5))
def heat_kernel(y, s, x, thresh, n_max, sde):
    s = jnp.ones((x.shape[0], 1)) * s
    marginal_log_prob = partial(sde.marginal_log_prob, thresh=thresh, n_max=n_max)
    return jnp.exp(jax.vmap(marginal_log_prob)(x, y, s))


@partial(jax.jit, static_argnums=(5))
def brownian_motion_traj(previous_x, N, dt, timesteps, traj, sde):
    rng = jax.random.PRNGKey(0)

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


@partial(jax.jit, static_argnums=(4))
def brownian_motion(previous_x, N, dt, timesteps, sde):
    rng = jax.random.PRNGKey(0)

    def body(step, val):
        rng, x = val
        t = jnp.broadcast_to(timesteps[step], (x.shape[0], 1))
        rng, z = sde.manifold.random_normal_tangent(
            state=rng, base_point=x, n_samples=x.shape[0]
        )
        drift, diffusion = sde.coefficients(x, t)  # sde.sde(x, t)
        tangent_vector = drift * dt + batch_mul(diffusion, jnp.sqrt(dt) * z)
        x = sde.manifold.metric.exp(tangent_vec=tangent_vector, base_point=x)
        return rng, x

    _, x = jax.lax.fori_loop(0, N, body, (rng, previous_x))
    return x


### S2


def vmf_kde(ys, kappa):
    def kernel(x):
        probs = vMF_pdf(x, ys, kappa)
        return jnp.mean(probs, axis=1)

    return kernel


def spherical_to_cartesian(theta, phi):
    x = jnp.concatenate(
        [jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)],
        axis=-1,
    )
    return x


def test_s2_normalisation():
    S2 = Hypersphere(dim=2)
    x0 = jnp.expand_dims(jnp.array([1.0, 0.0, 0.0]), 0)
    kernel = partial(heat_kernel, x=x0, manifold=S2, thresh=0.05, n_max=10)
    # kernel = partial(heat_kernel, x=x0, manifold=S2, thresh=jnp.inf, n_max=1)

    K = 5000
    eps = 0.0  # 1e-3
    theta = jnp.linspace(eps, jnp.pi - eps, K)
    phi = jnp.linspace(eps, 2 * jnp.pi - eps, K)
    theta, phi = jnp.meshgrid(theta, phi)
    theta = theta.reshape(-1, 1)
    phi = phi.reshape(-1, 1)
    x = spherical_to_cartesian(theta, phi)
    volume = jnp.pi * jnp.pi
    lambda_x = jnp.sin(theta)

    # ts = jnp.linspace(0, 0.02, 10+1)[1:]
    ts = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 5.0]
    volume = (2 * np.pi) * np.pi

    for t in ts:
        pdf = kernel(x, t)
        Z = jnp.mean(batch_mul(pdf, lambda_x) * volume)
        print(f"t={t:.3f} | Z={Z:.2f} | mean={pdf.mean():.2f} | std={pdf.std():.2f}")
        plot_and_save2(
            [], pdf=partial(kernel, s=t), out=f"images/s2_pdf/hk_{t:.3f}.jpg"
        )

    # ts = jnp.linspace(0, 1., 100+1)[1:]
    # heatmaps = [partial(kernel, s=t) for t in ts]
    # plot_and_save_video2(heatmaps, fps=10, out="forward.mp4")


def test_s2():
    M = 500
    S2 = Hypersphere(dim=2)
    x0 = jnp.expand_dims(jnp.array([1.0, 0.0, 0.0]), 0)
    x0b = jnp.repeat(x0, 2 * 500, 0)
    sde = Brownian(S2, tf=3, beta_0=1, beta_f=1)
    # sde = Brownian(S2, tf=1, beta_0=0.1, beta_f=3)
    # sde = Brownian(S2, tf=3, beta_0=1., beta_f=1)
    kernel = partial(heat_kernel, sde=sde)
    S2_brownian_motion = partial(brownian_motion, sde=sde)

    # ts = jnp.linspace(0, 1., 4+1)[1:]
    # ts = [0.002, 0.005, 0.01, 0.02, 0.05]
    # ts = [0.1, 0.2, 0.5, 1., 2., 5.]
    # ts = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    # ts = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ts = [0.01, 0.1, sde.tf]
    # eps = 0.
    eps = jnp.pi * 3 / 4
    r = jnp.linspace(0.0, jnp.pi - eps, M)
    r = jnp.concatenate([-jnp.flip(r), r]).reshape(-1, 1)
    M = 2 * M
    v0 = jnp.array([[0.0, 0.0, 1.0]]) * r
    # r = S2.metric.norm(v0)
    x = S2.metric.exp(v0, x0)

    probs_trunc_5 = jnp.array(
        [kernel(x=x0b, y=x, s=t, thresh=0.0, n_max=5) for t in ts]
    )
    probs_trunc_10 = jnp.array(
        [kernel(x=x0b, y=x, s=t, thresh=0.0, n_max=10) for t in ts]
    )
    # probs_trunc_20 = jnp.array([kernel(x=x0b, y=x, s=t, thresh=0., n_max=20) for t in ts])
    probs_trunc_50 = jnp.array(
        [kernel(x=x0b, y=x, s=t, thresh=0.0, n_max=50) for t in ts]
    )
    probs_dev = jnp.array(
        [kernel(x=x0b, y=x, s=t, thresh=jnp.inf, n_max=1) for t in ts]
    )

    ### Sample heat kernel + density estimation
    # K = 1000000
    # previous_x = jnp.repeat(x0, K, axis=0)
    # probs_samples = jnp.zeros((len(ts), M))
    # N = 100

    # for i, t in enumerate(ts):
    #     print(t)
    #     eps = 1e-3
    #     dt = (t - eps) / N
    #     timesteps = jnp.linspace(eps, t, N)
    #     y = S2_brownian_motion(previous_x, N, dt, timesteps)
    #     kde = vmf_kde(y, kappa=50 / t)
    #     prob = kde(x)
    #     probs_samples = probs_samples.at[i].set(prob)

    probs = jnp.concatenate(
        list(
            map(
                lambda x: jnp.expand_dims(x, 1),
                [
                    probs_trunc_5,
                    probs_trunc_10,
                    probs_trunc_50,
                    # probs_samples,
                    probs_dev,
                ],
            )
        ),
        axis=1,
    )
    dim_0_names = list(map(lambda x: f"t={x}", ts))
    dim_1_names = [
        "Sturm-Liouville (5)",
        "Sturm-Liouville (10)",
        "Sturm-Liouville (50)",
        # "samples",
        "Varadhan expansion",
    ]
    plot(
        r,
        probs,
        dim_0_names,
        dim_1_names,
        # out=f"images/arxiv/s2_heat_kernel_beta_{sde.tf}_{sde.beta_0:.1f}_{sde.beta_f:.1f}",
        out=f"images/arxiv/s2_heat_kernel",
    )


### S1


def plot_s1(x, ys, dim_0_names, dim_1_names, out):
    K, J, _ = ys.shape
    projection = "polar"
    fig, axis = plt.subplots(
        nrows=1,
        ncols=K,
        figsize=(12, 5),
        sharex=True,
        sharey=False,
        subplot_kw={"projection": projection},
    )
    colours = sns.cubehelix_palette(n_colors=J, reverse=False)
    for k in range(K):
        for j in range(J):
            linestyle = "--" if j == J - 1 else "-"
            # print(ys[k, j].sum() * 2 * np.pi / N)
            axis[k].plot(
                x,
                ys[k, j],
                color=colours[j],
                alpha=0.5,
                label=dim_1_names[j],
                linestyle=linestyle,
            )
        axis[k].set_title(dim_0_names[k])
    # axis[-1].legend(loc='best')
    plt.savefig("{}.png".format(out), dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close("all")


def test_s1():
    S1 = Hypersphere(dim=1)
    x0 = jnp.expand_dims(jnp.array([1.0, 0.0]), 0)

    kernel = partial(heat_kernel, manifold=S1)

    # ts = [0.002, 0.005, 0.01, 0.02, 0.05]
    ts = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
    eps = jnp.pi * 3 / 4
    # ts = [0.1, 0.5, 2., 5., 10.]
    # eps = 0.
    N = 500
    # r = jnp.linspace(0., 2 * jnp.pi - eps, N).reshape(-1, 1)
    r = jnp.linspace(0.0, jnp.pi - eps, N)
    N = 2 * N
    r = jnp.concatenate([-jnp.flip(r), r]).reshape(-1, 1)
    v0 = jnp.array([[0.0, 1.0]]) * r
    # r = S1.metric.norm(v0)
    x = S1.metric.exp(v0, x0)
    probs_trunc_1 = jnp.array([kernel(x=x0, y=x, s=t, thresh=0.0, n_max=0) for t in ts])
    probs_trunc_3 = jnp.array([kernel(x=x0, y=x, s=t, thresh=0.0, n_max=1) for t in ts])
    probs_trunc_10 = jnp.array(
        [kernel(x=x0, y=x, s=t, thresh=0.0, n_max=5) for t in ts]
    )
    probs_dev = jnp.array([kernel(x=x0, y=x, s=t, thresh=jnp.inf, n_max=1) for t in ts])

    ### Sample heat kernel + density estimation
    K = 100000
    previous_x = jnp.repeat(x0, K, axis=0)
    rng = jax.random.PRNGKey(0)
    probs_samples = jnp.zeros((len(ts), N))

    # S1_brownian_motion = partial(brownian_motion_traj, manifold=S1)
    # N, T = 5000, 5.
    # dt = T / N * 2
    # timesteps = jnp.linspace(0, T, N + 1)
    # previous_x = jnp.repeat(x0, K, axis=0)
    # traj = jnp.zeros((N + 1, previous_x.shape[0], S1.dim + 1))
    # _, traj = S1_brownian_motion(previous_x, traj, dt, N)

    for i, t in enumerate(ts):
        rng, step_rng = jax.random.split(rng)
        ys = S1.random_walk(step_rng, previous_x, t)
        # ys = traj[list(timesteps).index(t)]

        v0 = S1.metric.log(ys, x0)
        d = v0[..., 1]  # S1.metric.norm(v0)

        # Interpolation
        kernel = gaussian_kde(d, bw_method=0.1)
        prob = kernel(r.reshape(-1))
        probs_samples = probs_samples.at[i].set(prob)

    probs = jnp.concatenate(
        list(
            map(
                lambda x: jnp.expand_dims(x, 1),
                [
                    probs_trunc_1,
                    probs_trunc_3,
                    probs_trunc_10,
                    probs_samples,
                    probs_dev,
                ],
            )
        ),
        axis=1,
    )
    dim_0_names = list(map(lambda x: f"t={x}", ts))
    dim_1_names = [
        "truncated(1)",
        "truncated(3)",
        "truncated(10)",
        "samples",
        "developement",
    ]
    plot(r, probs, dim_0_names, dim_1_names, out="images/s1_heat_kernel")
    # plot_s1(r, probs, ts, dim_1_names, out="images/s1_heat_kernel")

    ### histogram
    # out = 'images/s1_samples'
    # density, bins = jnp.histogram(r, bins=20, density=True) # range=[0, jnp.pi]

    # K, J = 1, 1
    # projection = "polar"
    # fig, axis = plt.subplots(nrows=1, ncols=K, figsize=(12, 5), sharex=True, sharey=False, subplot_kw={'projection': projection})
    # colours = sns.cubehelix_palette(n_colors=J, reverse=False)
    # for k in range(K):
    #     for j in range(J):
    #         linestyle = '--' if j == J - 1 else '-'
    #         # axis[k].plot(x, ys[k, j], color=colours[j], alpha=0.5, label=dim_1_names[j], linestyle=linestyle)
    #         print(bins)
    #         print(density)
    #         axis.bar(bins[:-1], density, width=(2*np.pi)/len(density), color=colours[j], alpha=0.5, linestyle=linestyle, fill=False)
    #         axis.set_yticks([])
    # # axis[-1].legend(loc='best')
    # plt.savefig("{}.png".format(out), dpi=300, bbox_inches="tight")
    # plt.clf()
    # plt.close("all")


if __name__ == "__main__":
    # test_s2_normalisation()
    test_s2()
    # test_s1()
