import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import geomstats.backend as gs
from riemannian_score_sde.models.distribution import (
    WrapNormDistribution as WrappedNormal,
)


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


class WrapNormDistribution:
    def __init__(self, batch_dims, manifold, scale=1.0, mean=None, seed=0, rng=None):
        self.manifold = manifold
        self.batch_dims = batch_dims
        if mean is None:
            mean = jnp.zeros(manifold.dim)
        mean = jnp.array(mean)
        self.dist = WrappedNormal(manifold, scale, mean)
        self.rng = rng if rng is not None else jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        self.rng, rng = jax.random.split(self.rng)
        return self.dist.sample(rng, self.batch_dims), None


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
