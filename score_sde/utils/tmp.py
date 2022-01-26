import jax
from jax import numpy as jnp
import numpy as np
import geomstats.backend as gs


def log_prob(likelihood_fn, transform, x):
    rng = jax.random.PRNGKey(0)
    z = transform.inv(x)
    logp, zT, nfe = likelihood_fn(rng, z)
    logp -= transform.log_abs_det_jacobian(z, x)
    return logp, zT, nfe


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


def compute_normalization(likelihood_fn, transform, model_manifold, N=200, eps=0.):
    xs, theta, phi = get_spherical_grid(N, eps)

    logp, zT, nfe = log_prob(likelihood_fn, transform, xs)
    print("nfe", nfe)
    belongs_manifold = model_manifold.belongs(zT, atol=1e-4)
    print("belongs_manifold", jnp.sum(belongs_manifold) / belongs_manifold.shape[0])

    prob = jnp.exp(logp)
    volume = (2 * np.pi) * np.pi
    lambda_x = jnp.sin(theta).reshape((-1))
    Z = (prob * lambda_x).mean() * volume

    return Z.item()
