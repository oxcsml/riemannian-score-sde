import jax
import jax.numpy as jnp

from score_sde.sde import SDE, RSDE
from score_sde.sde import VPSDE as VPSDEBase, RSDE as RSDEBase
from score_sde.utils import batch_mul


class Brownian(SDE):
    def __init__(self, manifold, tf: float, t0: float = 0, beta_0=0.1, beta_f=20):
        """Construct a Brownian motion on a compact manifold"""
        super().__init__(tf, t0)
        self.beta_0 = beta_0
        self.beta_f = beta_f
        self.manifold = manifold

    def beta_t(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return self.beta_0 + normed_t * (self.beta_f - self.beta_0)

    def coefficients(self, x, t):
        beta_t = self.beta_t(t)
        drift = jnp.zeros_like(x)
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """Should not rely on closed-form marginal probability"""
        # TODO: this is a Euclidean approx
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_f - self.beta_0) - 0.5 * t * self.beta_0
        )
        # mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        return jnp.zeros_like(x), std

    def marginal_sample(self, rng, x, t):
        from score_sde.sampling import (
            EulerMaruyamaManifoldPredictor,
            get_pc_sampler,
        )  # TODO: remove from class

        perturbed_x = self.manifold.random_walk(rng, x, t)
        if perturbed_x is None:
            # TODO: should pmap the pc_sampler?
            sampler = get_pc_sampler(
                self,
                100,
                predictor="EulerMaruyamaManifoldPredictor",
                corrector=None,
            )
            perturbed_x, _ = sampler(rng, x, tf=t)
        return perturbed_x

    # def marginal_log_prob(self, x0, x, t, **kwargs):
    #     # TODO: Should indeed vmap?
    #     # NOTE: reshape: https://github.com/google/jax/issues/2303
    #     s = 2 * (0.25 * t ** 2 * (self.beta_f - self.beta_0) + 0.5 * t * self.beta_0)
    #     return jnp.reshape(self.manifold.log_heat_kernel(x0, x, s, **kwargs), ())

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        s = 2 * (0.25 * t ** 2 * (self.beta_f - self.beta_0) + 0.5 * t * self.beta_0)
        logp_grad = self.manifold.grad_marginal_log_prob(x0, x, s, **kwargs)
        return None, logp_grad

    def sample_limiting_distribution(self, rng, shape):
        return self.manifold.random_uniform(state=rng, n_samples=shape[0])

    def limiting_distribution_logp(self, z):
        return -jnp.ones([*z.shape[:-1]]) * self.manifold.metric.log_volume

    def reverse(self, score_fn):
        return ReverseBrownian(self, score_fn)


class ReverseBrownian(Brownian):
    """Reverse time SDE, assuming the drift coefficient is spatially homogenous"""

    def __init__(self, sde: SDE, score_fn):
        super().__init__(sde.manifold, tf=sde.t0, t0=sde.tf)

        self.sde = sde
        self.score_fn = score_fn

    def coefficients(self, x, t):
        forward_drift, diffusion = self.sde.coefficients(x, t)
        score_fn = self.score_fn(x, t)

        # compute G G^T score_fn
        if self.sde.full_diffusion_matrix:
            # if square matrix diffusion coeffs
            reverse_drift = forward_drift - jnp.einsum(
                "...ij,...kj,...k->...i", diffusion, diffusion, score_fn
            )
        else:
            # if scalar diffusion coeffs (i.e. no extra dims on the diffusion)
            reverse_drift = forward_drift - jnp.einsum(
                "...,...,...i->...i", diffusion, diffusion, score_fn
            )

        return reverse_drift, diffusion

    def reverse(self):
        return self.sde


class VPSDE(VPSDEBase):
    def __init__(self, manifold, tf: float, t0: float = 0, beta_0=0.1, beta_f=20):
        super().__init__(tf, t0, beta_0, beta_f)
        self.manifold = manifold

    def marginal_sample(self, rng, x, t):
        mean, std = self.marginal_prob(x, t)
        z = jax.random.normal(rng, x.shape)
        return mean + batch_mul(std, z)

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        mean, std = self.marginal_prob(x0, t)
        std = jnp.expand_dims(std, -1)
        score = - 1 / (std ** 2) * (x - mean)
        logp = None
        return logp, score

    def reverse(self, score_fn):
        return RSDE(self, score_fn)


class RSDE(RSDEBase):
    def __init__(self, sde: SDE, score_fn):
        super().__init__(sde, score_fn)
        self.manifold = sde.manifold
