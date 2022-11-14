# %%
%load_ext autoreload
%autoreload 2
import os
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from score_sde.utils import cfg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

os.environ['GEOMSTATS_BACKEND'] = 'jax'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# %%
import os
import socket
import logging
from functools import partial
from timeit import default_timer as timer

import jax
import optax
import numpy as np
import haiku as hk
from tqdm import tqdm
from jax import numpy as jnp
from score_sde.models.flow import SDEPushForward, MoserFlow
from geomstats.geometry.hypersphere import Hypersphere

# %%
with open("/data/ziz/not-backed-up/mhutchin/score-sde/scripts/blender/mesh_utils.py") as file:
    exec(file.read())
# %%

def m_to_e(M):
    phi = M[..., 0]
    theta = M[..., 1]

    return jnp.stack(
        [
            jnp.sin(phi) * jnp.cos(theta),
            jnp.sin(phi) * jnp.sin(theta),
            jnp.cos(phi),
        ],
        axis=-1,
    )


# S2 = EmbeddedS2(1.0)
import numpy as np
num_points = 30
phi = np.linspace(0, np.pi, num_points)
theta = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]
phi, theta = np.meshgrid(phi, theta, indexing="ij")
phi = phi.flatten()
theta = theta.flatten()
m = np.stack(
    [phi, theta], axis=-1
)  ### NOTE this ordering, I can change it but it'll be a pain, its latitude, longitude
density = 10
m_dense = np.stack(
    np.meshgrid(
        np.linspace(0, np.pi, density * num_points),
        np.linspace(0, 2 * np.pi, 2 * density * num_points + 1)[:-1],
        indexing="ij",
    ),
    axis=-1,
)
x = m_to_e(m)
verticies, faces = mesh_to_polyscope(
    x.reshape((num_points, num_points, 3)), wrap_y=False
)

uv = m / m.max(axis=0, keepdims=True)
mesh_obj = regular_square_mesh_to_obj(
    x.reshape((num_points, num_points, 3)), wrap_y=True
)

save_obj(mesh_obj, "data/sphere.obj")

# %%
sphere = Hypersphere(2)
x_dense = m_to_e(m_dense)
# %%
base_point = jnp.array([0,-1,0])

def wrap_prob(x, sigma):
    tv = sphere.log(x, base_point=base_point[None, :])
    dist = jnp.linalg.norm(tv, axis=-1) ** 2
    norm_pdf =  1/jnp.sqrt(2*jnp.pi*sigma**2) * jnp.exp(- 0.5 * dist / sigma**2)
    logdetexp = sphere.metric.logdetexp(base_point[None, :], x)
    return norm_pdf + logdetexp

def wrap_logprob(x, sigma):
    tv = sphere.log(x, base_point=base_point[None, :])
    dist = jnp.linalg.norm(tv, axis=-1) ** 2
    norm_pdf =  - 0.5 * dist / sigma**2
    logdetexp = sphere.metric.logdetexp(base_point[None, :], x)
    return jnp.exp(norm_pdf + logdetexp)

def rnorm_logprob(x, sigma):
    tv = sphere.log(x, base_point=base_point[None, :])
    dist = jnp.linalg.norm(tv, axis=-1) ** 2
    norm_pdf =  - 0.5 * dist / sigma**2
    return jnp.exp(norm_pdf)

# %%
import geometric_kernels
# Import a space and an appropriate kernel.
from geometric_kernels.spaces.hypersphere import Hypersphere as Hypersphere2
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel

# Create a manifold (2-dim sphere).
hypersphere = Hypersphere2(dim=2)

kernel = MaternKarhunenLoeveKernel(hypersphere, 1000)
params, state = kernel.init_params_and_state()
params["nu"] = np.inf
params["lengthscale"] = np.array([1.])

def kernel_prob(x, sigma):
    params["lengthscale"] = np.array([sigma])
    return kernel.K(params, state, np.array(x), np.array(base_point[None, :]))


# %%

sigmas = np.power(10,np.linspace(np.log10(0.08), 0.7, 30))

for sigma in sigmas:
    make_scalar_texture(
        partial(wrap_prob, sigma=sigma), 
        m_to_e(m_dense),
        # m_to_e(m_dense).swapaxes(0,1),
        f"data/exp_pushforward_{sigma:0.2f}.png",
        cmap='viridis',
    )

    make_scalar_texture(
        partial(kernel_prob, sigma=sigma), 
        m_to_e(m_dense),
        # m_to_e(m_dense).swapaxes(0,1),
        f"data/brownian_{sigma:0.2f}.png",
        cmap='viridis',
    )
# %%

from riemannian_score_sde.models.distribution import WrapNormDistribution

wrap_norm = WrapNormDistribution(Hypersphere(2))

# wrap_norm.log_prob(
#     x_dense
# )

make_scalar_texture(
    partial(wrap_logprob, sigma=0.5), 
    x_dense,
    # m_to_e(m_dense).swapaxes(0,1),
    f"data/wrap_norm.png",
    cmap='viridis',
)

make_scalar_texture(
    partial(rnorm_logprob, sigma=0.5), 
    x_dense,
    # m_to_e(m_dense).swapaxes(0,1),
    f"data/rnorm.png",
    cmap='viridis',
)

make_scalar_texture(
    lambda x: jnp.zeros_like(x[..., 0]), 
    x_dense,
    # m_to_e(m_dense).swapaxes(0,1),
    f"data/unif.png",
    cmap='viridis',
)
# %%
