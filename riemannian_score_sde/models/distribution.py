import numpy as np
import jax.numpy as jnp

from geomstats.geometry.euclidean import Euclidean
from score_sde.sde import SDE
from distrax import MultivariateNormalDiag


class UniformDistribution:
    """Uniform density on compact manifold"""

    def __init__(self, manifold, **kwargs):
        self.manifold = manifold

    def sample(self, rng, shape):
        return self.manifold.random_uniform(state=rng, n_samples=shape[0])

    def log_prob(self, z):
        return -jnp.ones([z.shape[0]]) * self.manifold.log_volume

    def grad_U(self, x):
        return jnp.zeros_like(x)


class MultivariateNormal(MultivariateNormalDiag):
    def __init__(self, dim, mean=None, scale=None, **kwargs):
        mean = jnp.zeros((dim)) if mean is None else mean
        scale = jnp.ones((dim)) if scale is None else scale
        super().__init__(mean, scale)

    def sample(self, rng, shape):
        return super().sample(seed=rng, sample_shape=shape)

    def log_prob(self, z):
        return super().log_prob(z)

    def grad_U(self, x):
        return x / (self.scale_diag**2)


class DefaultDistribution:
    def __new__(cls, manifold, flow, **kwargs):
        if isinstance(flow, SDE):
            return flow.limiting
        else:
            if isinstance(manifold, Euclidean):
                zeros = jnp.zeros((manifold.dim))
                ones = jnp.ones((manifold.dim))
                return MultivariateNormalDiag(zeros, ones)
            elif hasattr(manifold, "random_uniform"):
                return UniformDistribution(manifold)
            else:
                # TODO: WrappedNormal
                raise NotImplementedError(f"No default distribution for {manifold}")


class WrapNormDistribution:
    def __init__(self, manifold, scale=1.0, mean=None):
        self.manifold = manifold
        if mean is None:
            mean = self.manifold.identity
        self.mean = mean
        # NOTE: assuming diagonal scale
        self.scale = (
            jnp.ones((mean.shape)) * scale
            if isinstance(scale, float)
            else jnp.array(scale)
        )

    def sample(self, rng, shape):
        mean = self.mean[None, ...]
        tangent_vec = self.manifold.random_normal_tangent(
            rng, self.manifold.identity, np.prod(shape)
        )[1]
        # tangent_vec = self.manifold.random_normal_tangent(rng, mean, np.prod(shape))[1]
        tangent_vec *= self.scale
        tangent_vec = self.manifold.metric.transpfrom0(mean, tangent_vec)
        return self.manifold.metric.exp(tangent_vec, mean)

    def log_prob(self, z):
        tangent_vec = self.manifold.metric.log(z, self.mean)
        tangent_vec = self.manifold.metric.transpback0(self.mean, tangent_vec)
        zero = jnp.zeros((self.manifold.dim))
        # TODO: to refactor axis contenation / removal
        if self.scale.shape[-1] == self.manifold.dim:  # poincare
            scale = self.scale
        else:  # hyperboloid
            scale = self.scale[..., 1:]
        norm_pdf = MultivariateNormalDiag(zero, scale).log_prob(tangent_vec)
        logdetexp = self.manifold.metric.logdetexp(self.mean, z)
        return norm_pdf - logdetexp

    def grad_U(self, x):
        def U(x):
            sq_dist = self.manifold.metric.dist(x, self.mean) ** 2
            res = 0.5 * sq_dist / (self.scale[0] ** 2)  # scale must be isotropic
            logdetexp = self.manifold.metric.logdetexp(self.mean, x)
            return res + logdetexp

        # U = lambda x: -self.log_prob(x)  #NOTE: this does not work

        return self.manifold.to_tangent(self.manifold.metric.grad(U)(x), x)
