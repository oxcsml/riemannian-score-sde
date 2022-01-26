import math
import jax
from jax import numpy as jnp
import numpy as np
from score_sde.utils.tmp import get_spherical_grid
from score_sde.datasets.unimodal import vMFDataset
from geomstats.geometry.hypersphere import Hypersphere
# from jax.scipy.special import gammaln


class PowerSpherical:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def log_prob(self, value):
        return self.log_normalizer() + self.scale * jnp.log1p(
            (self.loc * value).sum(-1)
        )

    def log_normalizer(self):
        dim = 2 + 1
        alpha = (dim - 1) / 2 + self.scale
        beta = (dim - 1) / 2
        return -(
            (alpha + beta) * math.log(2)
            + math.lgamma(alpha)
            - math.lgamma(alpha + beta)
            + beta * math.log(math.pi)
        )


N=1000
eps=1e-3
xs, theta, phi = get_spherical_grid(N, eps)

manifold = Hypersphere(2)
mu = [1., 0., 0.]
kappa = 100
distribution = vMFDataset(None, None, manifold, mu, kappa)
# distribution = PowerSpherical(mu, kappa)
logp = distribution.log_prob(xs)
# logp = jnp.ones(xs.shape[0]) * -manifold.metric.log_volume

prob = jnp.exp(logp)
volume = (2 * np.pi) * np.pi
lambda_x = jnp.sin(theta).reshape(prob.shape)
Z = (prob * lambda_x).mean() * volume

print(Z.item())
