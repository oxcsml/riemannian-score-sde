from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
import geomstats.backend as gs

from riemannian_score_sde.models.distribution import (
    WrapNormDistribution as WrappedNormal,
)


class vMFMixture:
    def __init__(
        self, batch_dims, rng, manifold, mu, kappa, weights=[0.5, 0.5], **kwargs
    ):
        self.manifold = manifold
        self.mu = jnp.array(mu)
        self.kappa = jnp.expand_dims(jnp.array(kappa), -1)
        self.weights = jnp.array(weights)
        self.batch_dims = batch_dims
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        rng = jax.random.split(self.rng, num=3)

        self.rng = rng[0]
        choice_key = rng[1]
        normal_key = rng[2]

        indices = jax.random.choice(
            choice_key, a=len(self.weights), shape=self.batch_dims, p=self.weights
        )
        random_von_mises_fisher = jax.vmap(
            partial(
                self.manifold.random_von_mises_fisher,
                n_samples=np.prod(self.batch_dims),
            )
        )
        samples = random_von_mises_fisher(mu=self.mu[indices], kappa=self.kappa[indices])
        diag = jnp.diag_indices(np.prod(self.batch_dims))
        samples = samples[diag]
        return (samples, None)


class WrapNormMixtureDistribution:
    def __init__(
        self,
        batch_dims,
        manifold,
        mean,
        scale,
        seed=0,
        rng=None,
    ):
        self.mean = jnp.array(mean)
        self.K = self.mean.shape[0]
        self.scale = jnp.array(scale)
        self.batch_dims = batch_dims
        self.manifold = manifold
        self.rng = rng if rng is not None else jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        n_samples = np.prod(self.batch_dims)
        ks = jnp.arange(self.K)
        self.rng, next_rng = jax.random.split(self.rng)
        _, k = gs.random.choice(state=next_rng, a=ks, n=n_samples)
        mean = self.mean[k]
        scale = self.scale[k]
        tangent_vec = self.manifold.random_normal_tangent(
            next_rng, self.manifold.identity, n_samples
        )[1]
        tangent_vec *= scale
        tangent_vec = self.manifold.metric.transpfrom0(mean, tangent_vec)
        samples = self.manifold.metric.exp(tangent_vec, mean)
        return (samples, None)

    def log_prob(self, x):
        def component_log_prob(mean, scale):
            return WrappedNormal(self.manifold, scale, mean).log_prob(x)

        component_log_like = jax.vmap(component_log_prob)(self.mean, self.scale)
        b = 1 / self.K * jnp.ones_like(component_log_like)
        return logsumexp(component_log_like, axis=0, b=b)
