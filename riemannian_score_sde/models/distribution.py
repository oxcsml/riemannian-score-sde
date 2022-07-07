import jax.numpy as jnp

from geomstats.geometry.euclidean import Euclidean
from score_sde.models.distribution import NormalDistribution
from score_sde.sde import SDE


class UniformDistribution:
    """Uniform density on compact manifold"""

    def __init__(self, manifold, **kwargs):
        self.manifold = manifold

    def sample(self, rng, shape):
        return self.manifold.random_uniform(state=rng, n_samples=shape[0])

    def log_prob(self, z):
        return -jnp.ones([z.shape[0]]) * self.manifold.log_volume


class DefaultDistribution:
    def __new__(cls, manifold, flow, **kwargs):
        if isinstance(flow, SDE):
            return flow.limiting
        else:
            if isinstance(manifold, Euclidean):
                return NormalDistribution()
            elif hasattr(manifold, "random_uniform"):
                return UniformDistribution(manifold)
            else:
                # TODO: WrappedNormal
                raise NotImplementedError(f"No default distribution for {manifold}")
