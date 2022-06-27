
# %%
%load_ext autoreload
%autoreload 2

# %%
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
import jax
import jax.numpy as jnp
from geomstats.geometry.hypersphere import Hypersphere
from riemannian_score_sde.datasets.simple import Wrapped
from score_sde.utils.vis import latlon_from_cartesian, cartesian_from_latlong
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
