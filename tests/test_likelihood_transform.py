import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
import jax
import jax.numpy as jnp
from riemannian_score_sde.models.transform import *

rng = jax.random.PRNGKey(0)

# SO(3) & exponential map
print("SO(3)")
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from riemannian_score_sde.utils.normalization import compute_normalization
from score_sde.models import get_likelihood_fn_w_transform
import numpy as np

K = 4
manifold = SpecialOrthogonal(n=3, point_type="matrix")
rng, next_rng = jax.random.split(rng)
base_point = manifold.identity
# base_point = manifold.random_uniform(next_rng, n_samples=1)
transform = TanhExpMap(manifold)


class NormalDistribution:
    def __init__(self, mean=0.0, scale=1.0):
        super().__init__()
        self.mean = mean
        self.scale = scale

    def sample(self, rng, shape):
        return self.mean + self.scale * jax.random.normal(rng, shape)

    def log_prob(self, z):
        shape = z.shape
        d = np.prod(shape[1:])
        logp_fn = (
            lambda z: -d / 2.0 * jnp.log(2 * np.pi)
            - d * jnp.log(self.scale)
            - jnp.sum(((z - self.mean) / self.scale) ** 2) / 2.0
        )
        return jax.vmap(logp_fn)(z)


flow = transform
base = NormalDistribution(mean=0.0, scale=1.0)
likelihood_fn = lambda y, **kwargs: (base.log_prob(y), 0)

# Normalization constant of base distribition in R^3
N = 100
d = 3
bound = 5
xs = jnp.linspace(-bound, bound, N)
xs = d * [xs]
xs = jnp.meshgrid(*xs)
xs = jnp.concatenate([x.reshape(-1, 1) for x in xs], axis=-1)
logp = likelihood_fn(xs)[0]
prob = jnp.exp(logp)
Z = (prob).mean() * ((2 * bound) ** d)
print(f"Z = {Z:.2f}")

# Normalization constant of pushforward in SO(3)
likelihood_fn = get_likelihood_fn_w_transform(likelihood_fn, transform)
Z = compute_normalization(likelihood_fn, manifold)
print(f"Z = {Z:.2f}")
