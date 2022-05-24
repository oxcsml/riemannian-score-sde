import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
from functools import partial
from geomstats.geometry.hypersphere import Hypersphere
import jax
import jax.numpy as jnp
import geomstats.backend as gs
from score_sde.models.transform import *
from riemannian_score_sde.datasets.simple import Wrapped

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
# base_point = None
base_point = jnp.array([0.0, 1.0, 0.0])
transform = ExpMap(manifold, base_point)
rng, next_rng = jax.random.split(rng)
rng, x = manifold.random_normal_tangent(
    state=rng, base_point=transform.base_point, n_samples=K
)
# print('x', x)
# x = 0.1 * x
y = transform(x)
# print('y', y)
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

# ## Exponential map with radial tanh transform
# # base_point = jnp.array([0., 1., 0.])
# base_point = None
# transform = TanhExpMap(manifold, base_point)
# rng, next_rng = jax.random.split(rng)
# rng, x = manifold.random_normal_tangent(
#     state=rng, base_point=transform.base_point, n_samples=K
# )
# # x = 2 * x
# y = transform(x)
# x_prime = transform.inv(y)
# assert jnp.isclose(x, x_prime).all()
# jac = jax.vmap(jax.jacrev(transform))(x)
# # logdet_numerical = jnp.log(jnp.sqrt(jnp.linalg.det(jac.transpose((0, 2, 1)) @ jac)))
# logdet_numerical = rec_log_det(jac)
# logdet = transform.log_abs_det_jacobian(x, y)
# # print(logdet)
# # print(logdet_numerical)
# # assert jnp.isclose(logdet, logdet_numerical).all()


# SO(3) & exponential map
print("SO(3)")
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


def compute_approx_jacobian(points, vee, exp, log, inv, eps=0.01):
    points = jnp.expand_dims(points, -2)
    dim = points.shape[-1]
    basis = jnp.expand_dims(jnp.eye(dim), 0)
    group_delta = exp(basis * eps + points)
    points_inv = inv(exp(points))
    normal_coord = vee(log(points_inv @ group_delta))
    estimated_det_jac = jnp.linalg.det(normal_coord) / ((eps) ** dim)
    return estimated_det_jac


def approximate_so3_jacobian(points, eps):
    return compute_approx_jacobian(
        points, manifold.vee, manifold.exp, manifold.log, manifold.inverse, eps
    )


K = 4
# K = 10
manifold = SpecialOrthogonal(n=3, point_type="matrix")
rng, next_rng = jax.random.split(rng)

dataset = Wrapped(
    K=1,
    scale=0.1,
    mean="anti",
    conditional=False,
    batch_dims=(K,),
    manifold=manifold,
    seed=0,
)
P, _ = next(dataset)
print(manifold.metric.dist(P, dataset.mean))
A = gs.linalg.logm(P)
A_prime = manifold.log_from_identity(P)
assert jnp.isclose(A, A_prime).all()

base_point = manifold.identity
base_point = manifold.random_uniform(next_rng, n_samples=1)
# transform = ExpMap(manifold, base_point)
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
# print("P", P[0])
# print("P_prime", P_prime[0])
assert jnp.isclose(P, P_prime).all()
A = gs.linalg.logm(P)
A_prime = manifold.log_from_identity(P_prime)
# print("A", A[0])
# print("A_prime", A_prime[0])
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
# x = 0.2 * x
# x = 2.8 * x
# X = manifold.skew_matrix_from_vector(x)

Y = transform(x)
x_prime = transform.inv(Y)
# print(x)
# print(manifold.injectivity_radius)
# print(jnp.linalg.norm(x, axis=-1))
# print(x_prime)
assert jnp.isclose(x, x_prime).all()
func = lambda x: transform(x)
# func = lambda x: jax.scipy.linalg.expm(manifold.hat(x))
jac = jax.vmap(jax.jacrev(func))(x)
# print('jac', jac.shape)
inv_f_x = jnp.repeat(jnp.expand_dims(manifold.inverse(transform(x)), -1), 3, -1)
# print('inv_f_x', inv_f_x.shape)
jac = jax.vmap(manifold.compose, in_axes=-1, out_axes=-1)(inv_f_x, jac)
# print('jac', jac.shape)
is_tangent = jax.vmap(partial(manifold.is_tangent, atol=1e-2), in_axes=-1, out_axes=-1)(
    jac
)
# print('is_tangent', is_tangent.shape)
# print(is_tangent.all())
jac = jax.vmap(manifold.vee, in_axes=-1, out_axes=-1)(jac)
# print('jac', jac.shape)
logdet_numerical = sq_log_det(jac)
# print('logdet_numerical', logdet_numerical.shape)
# print(logdet_numerical)
logdet = transform.log_abs_det_jacobian(x, Y)
# print('logdet', logdet.shape)
# print(logdet)
assert jnp.isclose(logdet, logdet_numerical).all()
