# %%
%load_ext autoreload
%autoreload 2

# %%
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
import jax
import jax.numpy as jnp
from geomstats.geometry.hypersphere import Hypersphere
from riemannian_score_sde.datasets import Wrapped
from riemannian_score_sde.utils.vis import latlon_from_cartesian, cartesian_from_latlong
%matplotlib inline
import matplotlib.pyplot as plt
# %%
sphere = Hypersphere(dim=2)
dataset = Wrapped(8, 'random', 5, (10000,), sphere, 0, False, 'unif')

samples = next(dataset)[0]
samples_ = latlon_from_cartesian(samples)

xx, yy = jnp.meshgrid(
    jnp.linspace(-jnp.pi/2, jnp.pi/2, 100)[1:-1], jnp.linspace(-jnp.pi, jnp.pi, 100)[1:-1]
)

coords = jnp.stack([xx, yy], axis=-1)
coords_ = cartesian_from_latlong(coords)
coords = latlon_from_cartesian(coords_)
xx, yy = coords[..., 0], coords[..., 1]

prob = dataset.log_prob(coords_)

plt.contour(xx, yy, prob)
plt.scatter(samples_[...,0], samples_[...,1], s=0.1)
plt.scatter(latlon_from_cartesian(dataset.mean)[..., 0], latlon_from_cartesian(dataset.mean)[..., 1], c='r')
plt.gca().set_aspect('equal')

# %%
from riemannianvectorgp.utils import GlobalRNG, mesh_to_polyscope
from riemannianvectorgp.sparse_gp import SparseGaussianProcess
from riemannianvectorgp.manifold import EmbeddedS2
from riemannianvectorgp.kernel import (
    MaternCompactRiemannianManifoldKernel,
)
from riemannianvectorgp.utils import (
    mesh_to_obj,
    save_obj,
    make_faces_from_vectors,
    regular_square_mesh_to_obj,
    make_scalar_texture,
    square_mesh_to_obj,
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
        np.linspace(0, 2 * np.pi, density * num_points + 1)[1:],
        indexing="ij",
    ),
    axis=-1,
)
x = S2.m_to_e(m)
verticies, faces = mesh_to_polyscope(
    x.reshape((num_points, num_points, 3)), wrap_y=False
)

uv = m / m.max(axis=0, keepdims=True)
mesh_obj = regular_square_mesh_to_obj(
    x.reshape((num_points, num_points, 3)), wrap_y=True
)

save_obj(mesh_obj, "sphere_dense.obj")

# %%
x_dense = S2.m_to_e(m_dense)
# %%
x.reshape((num_points, num_points, 3)).swapaxes(0,1).reshape((-1, 3))
# %%
