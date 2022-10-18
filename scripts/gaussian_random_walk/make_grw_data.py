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
theta = np.linspace(0, 2 * np.pi, num_points + 1)[1:]
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
        np.linspace(0, 2 * np.pi, 2 * density * num_points + 1)[1:],
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

density=10
m_dense = np.stack(
    np.meshgrid(
        np.linspace(0, np.pi, density * num_points),
        (np.linspace(0, 2 * np.pi, 2 * density * num_points + 1)[1:]) % (2*np.pi),
        indexing="ij",
    ),
    axis=-1,
)

# %%
sphere = Hypersphere(2)


points = []
tangent_vecs = []
geodesics = []

points.append(jnp.array([0.0, -1.0, 0.0]))

rng = jax.random.PRNGKey(0)

length=10
step=0.1
k=10
n=int(length/step)
for i in range(n):
    rng, next_rng = jax.random.split(rng)
    tangent_vecs.append(step * sphere.random_normal_tangent(rng, points[-1])[1][0])
    geodesics.append(sphere.exp(jnp.linspace(0,1,k)[:, None] * tangent_vecs[-1][None, :], points[-1][None, :]))
    points.append(sphere.exp(tangent_vecs[-1], points[-1]))


for i in range(n):
    np.savetxt(f'data/point_{i}.csv', points[i], delimiter=',')
    np.savetxt(f'data/vec_{i}{i+1}.csv', jnp.concatenate([points[i], tangent_vecs[i]])[None, :], delimiter=',')
    np.savetxt(f'data/geodesic_{i}{i+1}.csv', geodesics[i], delimiter=',')

i=n
np.savetxt(f'data/point_{i}.csv', points[i], delimiter=',')

# %%
