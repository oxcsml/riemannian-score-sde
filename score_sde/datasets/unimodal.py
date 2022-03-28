import math
import jax.numpy as jnp
import numpy as np
from scipy.special import ive
from score_sde.utils import batch_mul


class vMFDataset:
    def __init__(
        self, batch_dims, rng, manifold, mu, kappa
    ):
        self.manifold = manifold
        self.d = self.manifold.dim + 1
        self.mu = jnp.array(mu)
        assert manifold.belongs(self.mu)
        self.kappa = jnp.array([kappa])
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

        return (samples, None)
        # return jnp.expand_dims(samples, axis=-1)

    def _log_normalization(self):
        output = -(
            (self.d / 2 - 1) * jnp.log(self.kappa)
            - (self.d / 2) * math.log(2 * math.pi)
            - (self.kappa + jnp.log(ive(self.d / 2 - 1, self.kappa)))
        )
        return output.reshape([1, *output.shape[:-1]])

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.kappa * (jnp.expand_dims(self.mu, 0) * x).sum(-1, keepdims=True)
        return output.reshape([*output.shape[:-1]])

    def entropy(self):
        output = -self.kappa * ive(self.d / 2, self.kappa) / ive((self.d / 2) - 1, self.kappa)
        return output.reshape([*output.shape[:-1]]) + self._log_normalization()


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
        return (samples, None)
