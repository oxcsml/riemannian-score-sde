import jax
import jax.numpy as jnp

from score_sde.sde import SDE, VPSDE as VPSDEBase, RSDE as RSDEBase
from score_sde.utils import batch_mul
from riemannian_score_sde.sampling import get_pc_sampler
from riemannian_score_sde.models.distribution import UniformDistribution


class Brownian(SDE):
    def __init__(self, manifold, beta_schedule, N=100):
        """Construct a Brownian motion on a compact manifold"""
        super().__init__(beta_schedule.tf, beta_schedule.t0)
        self.beta_schedule = beta_schedule
        self.manifold = manifold
        self.limiting = UniformDistribution(manifold)
        self.N = N

    def coefficients(self, x, t):
        beta_t = self.beta_schedule.beta_t(t)
        drift = jnp.zeros_like(x)
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """Should not rely on closed-form marginal probability"""
        # NOTE: this is a Euclidean approx!
        log_mean_coeff = self.beta_schedule.log_mean_coeff(t)
        # mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        return jnp.zeros_like(x), std

    def marginal_sample(self, rng, x, t, return_hist=False):
        out = self.manifold.random_walk(rng, x, self.beta_schedule.rescale_t(t))
        if return_hist or out is None:
            sampler = get_pc_sampler(
                self,
                self.N,
                predictor="GRW",
                return_hist=return_hist,
            )
            out = sampler(rng, x, tf=t)
        return out

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        s = self.beta_schedule.rescale_t(t)
        logp_grad = self.manifold.grad_marginal_log_prob(x0, x, s, **kwargs)
        return None, logp_grad

    def varhadan_exp(self, xs, xt, s, t):
        delta_t = self.beta_schedule.rescale_t(t) - self.beta_schedule.rescale_t(s)
        axis_to_expand = tuple(range(-1, -len(xt.shape), -1))  # (-1) or (-1, -2)
        delta_t = jnp.expand_dims(delta_t, axis=axis_to_expand)
        grad = self.manifold.log(xs, xt) / delta_t
        return delta_t, grad

    def reverse(self, score_fn):
        return RSDE(self, score_fn)


class VPSDE(VPSDEBase):
    def __init__(self, beta_schedule, manifold=None):
        super().__init__(beta_schedule)
        self.manifold = manifold

    def marginal_sample(self, rng, x, t):
        mean, std = self.marginal_prob(x, t)
        z = jax.random.normal(rng, x.shape)
        return mean + batch_mul(std, z)

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        mean, std = self.marginal_prob(x0, t)
        std = jnp.expand_dims(std, -1)
        score = -1 / (std**2) * (x - mean)
        logp = None
        return logp, score

    def reverse(self, score_fn):
        return RSDE(self, score_fn)


class RSDE(RSDEBase):
    def __init__(self, sde: SDE, score_fn):
        super().__init__(sde, score_fn)
        self.manifold = sde.manifold
