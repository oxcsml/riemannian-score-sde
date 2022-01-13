import jax.numpy as jnp
import numpy as np


class vMFDataset:
    def __init__(
        self, batch_dims, rng, manifold, mu, kappa
    ):
        self.manifold = manifold
        self.mu = jnp.array(mu)
        assert manifold.belongs(mu)
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


class DiracDataset:
    def __init__(
        self, batch_dims, mu, **kwargs
    ):
        self.mu = jnp.array(mu)
        self.batch_dims = jnp.array(batch_dims)

    def __iter__(self):
        return self

    def __next__(self):
        samples = jnp.repeat(self.mu.reshape(1, -1), self.batch_dims, 0)
        return samples
