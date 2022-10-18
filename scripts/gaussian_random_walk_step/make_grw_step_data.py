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

# %%

save_obj(mesh_obj, "data/sphere.obj")

# %%
sphere = Hypersphere(2)

base_point = jnp.array([0,-1,0])
ts_base_x = jnp.array([1,0,0])
ts_base_z = jnp.array([0,0,1])
sigma = 1.0

grid = jnp.meshgrid(
    jnp.linspace(-1,1,num_points*density),jnp.linspace(-1,1,num_points*density)
)

grid = grid[0][..., None] * ts_base_x + grid[1][..., None] * ts_base_z

def prob(x, sigma):
    dist = jnp.linalg.norm(x, axis=-1) ** 2
    return 1/jnp.sqrt(2*jnp.pi*sigma**2) * jnp.exp(- dist / sigma**2)

make_scalar_texture(
    partial(prob, sigma=sigma), 
    grid,
    f"data/plane_texture.png",
    cmap='viridis',
)

# %%

tv = jnp.array([0.6,0,0])
geodesic = sphere.exp(jnp.linspace(0,1,100)[:, None] * tv[None, :], base_point[None, :])


np.savetxt(f'data/point.csv', geodesic[-1], delimiter=',')
np.savetxt(f'data/vec.csv', jnp.concatenate([base_point, tv])[None, :], delimiter=',')
np.savetxt(f'data/geodesic.csv', geodesic, delimiter=',')

# i=n
# np.savetxt(f'data/point_{i}.csv', points[i], delimiter=',')
# %%
