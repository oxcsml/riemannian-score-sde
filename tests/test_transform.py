import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
from functools import partial
from geomstats.geometry.hypersphere import Hypersphere
import jax
import jax.numpy as jnp
import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from riemannian_score_sde.models.transform import *
from riemannian_score_sde.datasets import Wrapped

sq_log_det = lambda jac: jnp.log(jnp.abs(jnp.linalg.det(jac)))
rec_log_det = lambda jac: jnp.log(
    jnp.sqrt(jnp.linalg.det(jac.transpose((0, 2, 1)) @ jac))
)

rng = jax.random.PRNGKey(0)

## Inverse of stereographic map: R^d -> S^d
print("S^2")

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
logdet_numerical = rec_log_det(jac)
logdet = transform.log_abs_det_jacobian(x, y)
assert jnp.isclose(logdet, logdet_numerical).all()

# Exponential map
base_point = jnp.array([0.0, 1.0, 0.0])
transform = ExpMap(manifold, base_point)
rng, next_rng = jax.random.split(rng)
rng, x = manifold.random_normal_tangent(
    state=rng, base_point=transform.base_point, n_samples=K
)
y = transform(x)
x_prime = transform.inv(y)
assert jnp.isclose(x, x_prime).all()

## Radial tanh
print("tanh")
# transform = RadialTanhTransform(1., manifold.dim)
transform = RadialTanhTransform(manifold.injectivity_radius, manifold.dim)
rng, next_rng = jax.random.split(rng)
x = jax.random.normal(next_rng, shape=[K, 1])
x = 2 * x
y = transform(x)
x_prime = transform.inv(y)
assert jnp.isclose(x, x_prime).all()
jac = jax.vmap(jax.jacrev(transform))(x)
logdet_numerical = sq_log_det(jac)
logdet = transform.log_abs_det_jacobian(x, y)
assert jnp.isclose(logdet, logdet_numerical).all()

# SO(3) & exponential map
print("SO(3)")

K = 4
# K = 10
manifold = SpecialOrthogonal(n=3, point_type="matrix")
rng, next_rng = jax.random.split(rng)

dataset = Wrapped(
    K=1,
    scale=0.1,
    scale_type="random",
    mean="anti",
    conditional=False,
    batch_dims=(K,),
    manifold=manifold,
    seed=0,
)
P, _ = next(dataset)
A = gs.linalg.logm(P)
A_prime = manifold.log_from_identity(P)
assert jnp.isclose(A, A_prime).all()

base_point = manifold.identity
base_point = manifold.random_uniform(next_rng, n_samples=1)
radius = manifold.injectivity_radius
transform = TanhExpMap(manifold, radius=radius)
# transform = RadialTanhTransform(manifold.injectivity_radius, manifold.dim)
rng, next_rng = jax.random.split(rng)
_, X = manifold.random_normal_tangent(
    state=rng, base_point=transform.base_point, n_samples=K
)

# testing that exp and log map works
assert jnp.isclose(X, manifold.hat(manifold.vee(X))).all()
P = gs.linalg.expm(X)
P_prime = manifold.exp_from_identity(X)
assert jnp.isclose(P, P_prime).all()
A = gs.linalg.logm(P)
A_prime = manifold.log_from_identity(P_prime)
assert jnp.isclose(A, A_prime).all()
assert jnp.isclose(A, X).all()
assert jnp.isclose(
    manifold.metric.squared_norm(manifold.lie_algebra.basis_normed[..., 0]), 1.0
)

delta_ij = manifold.metric.inner_product(
    manifold.lie_algebra.basis_normed[..., 0], manifold.lie_algebra.basis_normed[..., 1]
)
delta_ii = manifold.metric.inner_product(
    manifold.lie_algebra.basis_normed[..., 0], manifold.lie_algebra.basis_normed[..., 0]
)
assert jnp.isclose(delta_ij, 0.0).all()
assert jnp.isclose(delta_ii, 1.0).all()

x = manifold.vector_from_skew_matrix(X)

Y = transform(x)
x_prime = transform.inv(Y)
assert jnp.isclose(x, x_prime).all()
func = lambda x: transform(x)
# func = lambda x: jax.scipy.linalg.expm(manifold.hat(x))
jac = jax.vmap(jax.jacrev(func))(x)
inv_f_x = jnp.repeat(jnp.expand_dims(manifold.inverse(transform(x)), -1), 3, -1)
jac = jax.vmap(manifold.compose, in_axes=-1, out_axes=-1)(inv_f_x, jac)
is_tangent = jax.vmap(partial(manifold.is_tangent, atol=1e-2), in_axes=-1, out_axes=-1)(
    jac
)
assert is_tangent.all()
jac = jax.vmap(manifold.vee, in_axes=-1, out_axes=-1)(jac)
logdet_numerical = sq_log_det(jac)
logdet = transform.log_abs_det_jacobian(x, Y)
assert jnp.isclose(logdet, logdet_numerical).all()
