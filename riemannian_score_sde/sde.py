import jax
import jax.numpy as jnp

from score_sde.sde import SDE, RSDE as RSDEBase, get_matrix_div_fn
from score_sde.models import get_score_fn
from score_sde.utils import batch_mul
from riemannian_score_sde.sampling import get_pc_sampler
from riemannian_score_sde.models.distribution import (
    UniformDistribution,
    MultivariateNormal,
    WrapNormDistribution,
)


class RSDE(RSDEBase):
    def __init__(self, sde: SDE, score_fn):
        super().__init__(sde, score_fn)
        self.manifold = sde.manifold


class Langevin(SDE):
    """Construct Langevin dynamics on a manifold"""

    def __init__(
        self,
        beta_schedule,
        manifold,
        ref_scale=0.5,
        ref_mean=None,
        N=100,
    ):
        super().__init__(beta_schedule)
        self.manifold = manifold
        self.limiting = WrapNormDistribution(manifold, scale=ref_scale, mean=ref_mean)
        self.N = N

    def drift(self, x, t):
        """dX_t =-0.5 beta(t) grad U(X_t)dt + sqrt(beta(t)) dB_t"""

        def fixed_grad(grad):
            is_nan_or_inf = jnp.isnan(grad) | (jnp.abs(grad) == jnp.inf)
            return jnp.where(is_nan_or_inf, jnp.zeros_like(grad), grad)

        drift_fn = jax.vmap(lambda x: -0.5 * fixed_grad(self.limiting.grad_U(x)))
        beta_t = self.beta_schedule.beta_t(t)
        drift = beta_t[..., None] * drift_fn(x)
        return drift

    def marginal_sample(self, rng, x, t, return_hist=False):
        out = self.manifold.random_walk(rng, x, self.beta_schedule.rescale_t(t))
        if return_hist or out is None:
            sampler = get_pc_sampler(
                self, self.N, predictor="GRW", return_hist=return_hist
            )
            out = sampler(rng, x, tf=t)
        return out

    def marginal_prob(self, x, t):
        # NOTE: this is only a proxy!
        log_mean_coeff = self.beta_schedule.log_mean_coeff(t)
        axis_to_expand = tuple(range(-1, -len(x.shape), -1))  # (-1) or (-1, -2)
        mean_coeff = jnp.expand_dims(jnp.exp(log_mean_coeff), axis=axis_to_expand)
        # mean = jnp.exp(log_mean_coeff)[..., None] * x
        mean = mean_coeff * x
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std

    def varhadan_exp(self, xs, xt, s, t):
        delta_t = self.beta_schedule.rescale_t(t) - self.beta_schedule.rescale_t(s)
        axis_to_expand = tuple(range(-1, -len(xt.shape), -1))  # (-1) or (-1, -2)
        delta_t = jnp.expand_dims(delta_t, axis=axis_to_expand)
        grad = self.manifold.log(xs, xt) / delta_t
        return delta_t, grad

    def reverse(self, score_fn):
        return RSDE(self, score_fn)


class VPSDE(Langevin):
    def __init__(self, beta_schedule, manifold=None, **kwargs):
        super().__init__(beta_schedule, manifold)
        self.limiting = MultivariateNormal(dim=manifold.dim)

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

    # def drift(self, x, t):
    #     beta_t = self.beta_schedule.beta_t(t)
    #     return -0.5 * beta_t[..., None] * x


class Brownian(Langevin):
    def __init__(self, manifold, beta_schedule, N=100):
        """Construct a Brownian motion on a compact manifold"""
        # super().__init__(beta_schedule, manifold, N=N)
        self.manifold = manifold
        self.limiting = UniformDistribution(manifold)
        self.N = N

        self.beta_schedule = beta_schedule
        self.tf = beta_schedule.tf
        self.t0 = beta_schedule.t0

    # def coefficients(self, x, t):
    #     beta_t = self.beta_schedule.beta_t(t)
    #     drift = jnp.zeros_like(x)
    #     diffusion = jnp.sqrt(beta_t)
    #     return drift, diffusion

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        s = self.beta_schedule.rescale_t(t)
        logp_grad = self.manifold.grad_marginal_log_prob(x0, x, s, **kwargs)
        return None, logp_grad

    def reparametrise_score_fn(self, score_fn, *args):
        return get_score_fn(self, score_fn, *args, std_trick=True, residual_trick=False)


class Hessian(Langevin):
    def __init__(self, manifold, beta_schedule, N=100):
        """Construct a Brownian motion on a compact manifold"""
        # NOTE: manifold must be convex set in R^d
        super().__init__(beta_schedule, manifold, N=N)
        # NOTE: only knows energy = 1/2 log det g
        # self.limiting = ???

    def drift(self, x, t):
        beta_t = self.beta_schedule.beta_t(t)
        G_inv = self.manifold.metric.metric_inverse_matrix  # NOTE: to implement
        div_term = get_matrix_div_fn(G_inv)(x, t, None)
        drift = -0.5 * beta_t[..., None] * div_term
        drift = drift / 2  # NOTE: due to normal coordinates change
        return drift

    def diffusion(self, x, t):
        beta_t = self.beta_schedule.beta_t(t)
        inv_G_sqrt = self.manifold.metric.metric_inverse_matrix_sqrt(x)
        diffusion = jnp.sqrt(beta_t)[..., :, :] * inv_G_sqrt  # NOTE: to implement
        return diffusion
