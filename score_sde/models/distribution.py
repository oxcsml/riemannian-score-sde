import jax
import jax.numpy as jnp
import numpy as np


class NormalDistribution:
    def __init__(self, **kwargs):
        pass

    def sample(self, rng, shape):
        return jax.random.normal(rng, shape)

    def log_prob(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logp_fn = lambda z: -N / 2.0 * jnp.log(2 * np.pi) - jnp.sum(z**2) / 2.0
        return jax.vmap(logp_fn)(z)
