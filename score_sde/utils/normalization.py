import jax
from jax import numpy as jnp
import numpy as np

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import _SpecialOrthogonalMatrices, _SpecialOrthogonal3Vectors


# def compute_microbatch_split(x, K=1):
#     """ Checks if batch needs to be broken down further to fit in memory. """
#     B = x.shape[0]
#     S = int(2e5 / (K * np.prod(x.shape[1:])))  # float heuristic for 12Gb cuda memory
#     return min(B, S)


# def compute_across_microbatch(func, x):
#     S = compute_microbatch_split(x)
#     split = jnp.split(x, S)
#     lw = jnp.concatenate([func(_x) for _x in split], axis=0)  # concat on batch
#     return lw


def get_spherical_grid(N, eps=0.):
    theta = jnp.linspace(eps, jnp.pi - eps, N // 2)
    phi = jnp.linspace(eps, 2 * jnp.pi - eps, N)

    theta, phi = jnp.meshgrid(theta, phi)
    theta = theta.reshape(-1, 1)
    phi = phi.reshape(-1, 1)
    xs = jnp.concatenate([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi), 
        jnp.cos(theta)
    ], axis=-1)
    volume = (2 * np.pi) * np.pi
    lambda_x = jnp.sin(theta).reshape((-1))
    return xs, volume, lambda_x


def get_so3_grid(N, eps=0.):
    print('get_so3_grid', N, eps)
    angle1 = jnp.linspace(-jnp.pi + eps, jnp.pi - eps, N)
    angle2 = jnp.linspace(-jnp.pi/2 + eps, jnp.pi/2 - eps, N // 2)
    angle3 = jnp.linspace(-jnp.pi + eps, jnp.pi - eps, N)

    angle1, angle2, angle3 = jnp.meshgrid(angle1, angle2, angle3)
    xs = jnp.concatenate([
        angle1.reshape(-1, 1),
        angle2.reshape(-1, 1),
        angle3.reshape(-1, 1),
    ], axis=-1)
    xs = jax.vmap(_SpecialOrthogonal3Vectors().matrix_from_tait_bryan_angles)(xs)
    
    # remove points too close from the antipole
    vs = jax.vmap(_SpecialOrthogonal3Vectors().rotation_vector_from_matrix)(xs)
    norm_v = jnp.linalg.norm(vs, axis=-1, keepdims=True)
    max_norm = jnp.pi - eps
    cond = jnp.expand_dims(norm_v <= max_norm, -1)
    # print('cond', cond.shape, (1 - cond).sum().item() / xs.shape[0] * 100)
    rescaled_vs = vs * max_norm / norm_v
    rescaled_xs = jax.vmap(_SpecialOrthogonal3Vectors().matrix_from_rotation_vector)(rescaled_vs)
    xs = jnp.where(cond, xs, rescaled_xs)

    volume = (2 * np.pi) * (2 * np.pi) * np.pi
    lambda_x = (jnp.sin(angle2 + np.pi/2)).reshape((-1))
    return xs, volume, lambda_x


def get_euclidean_grid(N, dim):
    dim = int(dim)
    bound = 10
    x = jnp.linspace(-bound, bound, N)
    xs = dim * [x]

    xs = jnp.meshgrid(*xs)
    xs = jnp.concatenate([x.reshape(-1, 1) for x in xs], axis=-1)
    volume = (2 * bound) ** dim
    lambda_x = (jnp.ones((xs.shape[0], 1))).reshape(-1)
    return xs, volume, lambda_x


def compute_normalization(likelihood_fn, manifold, context=None, N=None, eps=0.):
    if isinstance(manifold, Euclidean):
        N = N if N is not None else int(jnp.power(1e5, 1/manifold.dim))
        xs, volume, lambda_x = get_euclidean_grid(N, manifold.dim)
    elif isinstance(manifold, Hypersphere) and manifold.dim == 2:
        N = N if N is not None else 200
        xs, volume, lambda_x = get_spherical_grid(N, eps)
    elif isinstance(manifold, _SpecialOrthogonalMatrices) and manifold.dim == 3:
        N = N if N is not None else 50
        xs, volume, lambda_x = get_so3_grid(N, eps=1e-3)
    else:
        print('Only integration over R^d, S^2 and SO(3) is implemented.')
        return 0.

    if context is not None:
        context = jnp.repeat(jnp.expand_dims(context, 0), xs.shape[0], 0)
    logp = likelihood_fn(xs, context)
    prob = jnp.exp(logp)
    Z = (prob * lambda_x).mean() * volume
    return Z.item()
