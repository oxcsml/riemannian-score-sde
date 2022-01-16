import os
os.environ['GEOMSTATS_BACKEND'] = 'jax'
from geomstats.geometry.hypersphere import Hypersphere
import jax
import jax.numpy as jnp

from score_sde.models.transform import *


rng = jax.random.PRNGKey(0)

## Inverse of stereographic map: R^d -> S^d
dim = 2
K = 4
manifold = Hypersphere(dim)
transform = InvStereographic(manifold)
rng, next_rng = jax.random.split(rng)
# rng, x = manifold.random_normal_tangent(state=rng, base_point=transform.base_point, n_samples=K)
x = jax.random.normal(next_rng, shape=[K, transform.domain.dim])
y = transform(x)
x_prime = transform.inv(y)
assert jnp.isclose(x, x_prime).all()
jac = jax.vmap(jax.jacrev(transform))(x)
logdet_numerical = jnp.log(jnp.sqrt(jnp.linalg.det(jac.transpose((0, 2, 1)) @ jac)))
logdet = transform.log_abs_det_jacobian(x, y)
assert jnp.isclose(logdet, logdet_numerical).all()

# Exponential map
base_point = jnp.array([0., 1., 0.])  # or None
transform = ExpMap(manifold, base_point)
rng, next_rng = jax.random.split(rng)
rng, x = manifold.random_normal_tangent(state=rng, base_point=transform.base_point, n_samples=K)
x = 0.1 * x
y = transform(x)
x_prime = transform.inv(y)
assert jnp.isclose(x, x_prime).all()

## Radial tanh
transform = RadialTanhTransform(manifold.injectivity_radius, manifold.dim)
rng, next_rng = jax.random.split(rng)
x = jax.random.normal(next_rng, shape=[K, 3])
# rng, x = manifold.random_normal_tangent(state=rng, base_point=manifold.identity, n_samples=K)
x = 5 * x
y = transform(x)
x_prime = transform.inv(y)
assert jnp.isclose(x, x_prime).all()
jac = jax.vmap(jax.jacrev(transform))(x)
logdet_numerical = jnp.log(jnp.abs(jnp.linalg.det(jac)))
print('logdet_numerical', logdet_numerical.shape)
print(logdet_numerical)
logdet = transform.log_abs_det_jacobian(x, y)
print('logdet', logdet.shape)
print(logdet)
assert jnp.isclose(logdet, logdet_numerical).all()

## Exponential map with radial tanh transform
# base_point = jnp.array([0., 1., 0.])
base_point = None
transform = TanhExpMap(manifold, base_point)
rng, next_rng = jax.random.split(rng)
rng, x = manifold.random_normal_tangent(state=rng, base_point=transform.base_point, n_samples=K)
# x = 2 * x
y = transform(x)
x_prime = transform.inv(y)
assert jnp.isclose(x, x_prime).all()