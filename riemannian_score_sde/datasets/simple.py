import jax
import jax.numpy as jnp
import numpy as np


class Uniform:
    def __init__(self, batch_dims, manifold, seed, **kwargs):
        self.batch_dims = batch_dims
        self.manifold = manifold
        self.rng = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        rng, next_rng = jax.random.split(self.rng)
        self.rng = rng
        n_samples = np.prod(self.batch_dims)
        samples = self.manifold.random_uniform(state=next_rng, n_samples=n_samples)
        return (samples, None)
