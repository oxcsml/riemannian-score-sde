import math
from jax import numpy as jnp

from riemannian_score_sde.utils.normalization import compute_normalization
from riemannian_score_sde.datasets import vMFDataset
from geomstats.geometry.hypersphere import Hypersphere


class PowerSpherical:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def log_prob(self, value):
        return self.log_normalizer() + self.scale * jnp.log1p((self.loc * value).sum(-1))

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


print("S^2")
manifold = Hypersphere(2)
mu = [1.0, 0.0, 0.0]
kappa = 100
distribution = vMFDataset(None, None, manifold, mu, kappa)
# distribution = PowerSpherical(mu, kappa)

likelihood_fn = lambda y, *args, **kwargs: distribution.log_prob(y)

Z = compute_normalization(likelihood_fn, manifold, N=1000)
print(f"Z = {Z:.2f}")
