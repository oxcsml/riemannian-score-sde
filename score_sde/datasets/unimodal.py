import jax
import jax.numpy as jnp
import numpy as np
from score_sde.utils import batch_mul

class vMFDataset:
    def __init__(
        self, batch_dims, rng, manifold, mu, kappa
    ):
        self.manifold = manifold
        self.mu = jnp.array(mu)
        assert manifold.belongs(self.mu)
        self.kappa = kappa
        self.batch_dims = batch_dims
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        # rng = jax.random.split(self.rng)
        samples = self.manifold.random_von_mises_fisher(
            mu=self.mu,
            kappa=self.kappa,
            n_samples=np.prod(self.batch_dims)
        )
        samples = samples.reshape([*self.batch_dims, samples.shape[-1]])

        return samples
        # return jnp.expand_dims(samples, axis=-1)

    def log_prob(self, x):
        output = jnp.log(self.kappa) - jnp.log(2 * jnp.pi) - self.kappa - (1 - jnp.exp(- 2 * self.kappa))
        return output + self.kappa * (jnp.expand_dims(self.mu, 0) * x).sum(-1)


class DiracDataset:
    def __init__(
        self, batch_dims, mu, **kwargs
    ):
        self.mu = jnp.array(mu)
        self.batch_dims = batch_dims

    def __iter__(self):
        return self

    def __next__(self):
        n_samples=np.prod(self.batch_dims)
        samples = jnp.repeat(self.mu.reshape(1, -1), n_samples, 0)
        return samples
