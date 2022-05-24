import os
os.environ['GEOMSTATS_BACKEND'] = 'jax'
import jax
import jax.numpy as jnp
import geomstats.backend as gs

from score_sde.models.transform import *

sq_log_det = lambda jac: jnp.log(jnp.abs(jnp.linalg.det(jac)))
rec_log_det = lambda jac: jnp.log(jnp.sqrt(jnp.linalg.det(jac.transpose((0, 2, 1)) @ jac)))

rng = jax.random.PRNGKey(0)

# SO(3) & exponential map
print('SO(3)')
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from score_sde.utils.normalization import compute_normalization
import numpy as np

K = 4
manifold = SpecialOrthogonal(n=3, point_type="matrix")
rng, next_rng = jax.random.split(rng)
base_point = manifold.identity
# base_point = manifold.random_uniform(next_rng, n_samples=1) 
# radius = manifold.injectivity_radius
# transform = TanhExpMap(manifold, base_point, radius)
transform = TanhExpMap(manifold)
# transform = ExpMap(manifold, base_point)


class NormalDistribution:

    def __init__(self, mean=0., scale=1.):
        super().__init__()
        self.mean = mean
        self.scale = scale

    def sample(self, rng, shape):
        return self.mean + self.scale * jax.random.normal(rng, shape)

    def log_prob(self, z):
        shape = z.shape
        d = np.prod(shape[1:])
        logp_fn = lambda z: -d / 2.0 * jnp.log(2 * np.pi) - d * jnp.log(self.scale) - jnp.sum(((z - self.mean) / self.scale) ** 2) / 2.0 
        return jax.vmap(logp_fn)(z)


flow = transform
base = NormalDistribution(mean=0., scale=1.)
likelihood_fn = lambda rng, z: base.log_prob(z)

N = 100
d = 3
bound = 5
xs = jnp.linspace(-bound, bound, N)
xs = d * [xs]
xs = jnp.meshgrid(*xs)
xs = jnp.concatenate([x.reshape(-1, 1) for x in xs], axis=-1)
print(xs.shape)
logp = likelihood_fn(None, xs)
prob = jnp.exp(logp)
Z = (prob).mean() * ((2 * bound) ** d)
print(f"Z = {Z:.2f}")


# def get_likelihood_fn_w_transform(likelihood_fn, transform):
#     def log_prob(rng, R):
#         v = transform.inv(R)
        
#         # A = gs.linalg.logm(R)
#         # # A = jax.vmap(gs.linalg.logm)(R)
#         # A = jax.vmap(manifold.log_from_identity)(R)
#         # v = manifold.vee(A)

#         logp = likelihood_fn(rng, v)
        
#         trace_R = jnp.trace(R, axis1=-2, axis2=-1)
#         norm_v = jnp.linalg.norm(v, axis=-1)

#         # log_abs_det_jac = + jnp.log(3 - trace_R) - 2 * jnp.log(jnp.arccos((trace_R - 1) / 2))
#         log_abs_det_jac2 = + jnp.log(3 - trace_R) - 2 * jnp.log(norm_v)
        
#         log_abs_det_jac3 = -2 * jnp.log(norm_v) + jnp.log(2 - 2 * jnp.cos(norm_v))
        
#         log_abs_det_jac = transform.log_abs_det_jacobian(v, R)

#         # print(log_abs_det_jac)
#         # print(log_abs_det_jac2)
#         # print(log_abs_det_jac3)

#         logp -= log_abs_det_jac
#         return logp
#     return log_prob

from score_sde.models import get_likelihood_fn_w_transform

likelihood_fn = get_likelihood_fn_w_transform(likelihood_fn, transform)

Z = compute_normalization(likelihood_fn, manifold)
print(f"Z = {Z:.2f}")
