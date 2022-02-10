import jax
from jax import numpy as jnp
import numpy as np
from score_sde.models.transform import get_likelihood_fn_w_transform


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
    return xs, theta, phi


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


def compute_normalization(likelihood_fn, model_manifold=None, N=200, eps=0.):
    rng = jax.random.PRNGKey(0)
    xs, theta, phi = get_spherical_grid(N, eps)

    logp = likelihood_fn(rng, xs)

    prob = jnp.exp(logp)
    volume = (2 * np.pi) * np.pi
    lambda_x = jnp.sin(theta).reshape((-1))
    Z = (prob * lambda_x).mean() * volume

    return Z.item()
