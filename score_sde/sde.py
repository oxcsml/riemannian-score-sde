"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
Modified code from https://github.com/yang-song/score_sde
"""
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from score_sde.models.model import get_score_fn
from score_sde.utils import get_exact_div_fn
from score_sde.schedule import ConstantBetaSchedule


class SDE(ABC):
    # Specify if the sde returns full diffusion matrix, or just a scalar indicating diagonal variance

    full_diffusion_matrix = False
    spatial_dependent_diffusion = False

    def __init__(self, beta_schedule=ConstantBetaSchedule()):
        """Abstract definition of an SDE"""
        self.beta_schedule = beta_schedule
        self.tf = beta_schedule.tf
        self.t0 = beta_schedule.t0

    @abstractmethod
    def drift(self, x, t):
        """Compute the drift coefficients of the SDE at (x, t)

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        pass

    def diffusion(self, x, t):
        """Compute the diffusion coefficients of the SDE at (x, t)

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        beta_t = self.beta_schedule.beta_t(t)
        return jnp.sqrt(beta_t)

    def coefficients(self, x, t):
        """Compute the drift and diffusion coefficients of the SDE at (x, t)

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        return self.drift(x, t), self.diffusion(x, t)

    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x | x_0 = 0)$.

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at
        """
        raise NotImplementedError()

    def marginal_log_prob(self, x0, x, t):
        """Compute the log marginal distribution of the SDE, $log p_t(x | x_0 = 0)$.

        Parameters
        ----------
        x0: jnp.ndarray
            Location of the start of the diffusion
        x : jnp.ndarray
            Location of the end of the diffusion
        t : float
            Time of diffusion
        """
        raise NotImplementedError()

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        """Compute the log marginal distribution and its gradient

        Parameters
        ----------
        x0: jnp.ndarray
            Location of the start of the diffusion
        x : jnp.ndarray
            Location of the end of the diffusion
        t : float
            Time of diffusion
        """
        marginal_log_prob = lambda x0, x, t: self.marginal_log_prob(x0, x, t, **kwargs)
        logp_grad_fn = jax.value_and_grad(marginal_log_prob, argnums=1, has_aux=False)
        logp, logp_grad = jax.vmap(logp_grad_fn)(x0, x, t)
        logp_grad = self.manifold.to_tangent(logp_grad, x)
        return logp, logp_grad

    def sample_limiting_distribution(self, rng, shape):
        """Generate samples from the limiting distribution, $p_{t_f}(x)$.
        (distribution may not exist / be inexact)

        Parameters
        ----------
        rng : jnp.random.KeyArray
        shape : Tuple
            Shape of the samples to sample.
        """
        return self.limiting.sample(rng, shape)

    def limiting_distribution_logp(self, z):
        """Compute log-density of the limiting distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: limiting distribution sample
        Returns:
          log probability density
        """
        return self.limiting.log_prob(z)

    def discretize(self, x, t, dt):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probability flow sampling.
        Defaults to Euler-Maruyama discretization.

        Parameters
        ----------
        x : jnp.ndarray
            Location to evaluate the coefficients at
        t : float
            Time to evaluate the coefficients at

        Returns:
            f, G - the discretised SDE drift and diffusion coefficients
        """
        drift, diffusion = self.coefficients(x, t)
        f = drift * dt
        G = diffusion * jnp.sqrt(jnp.abs(dt))
        return f, G

    def reparametrise_score_fn(self, score_fn, *args):
        return get_score_fn(self, score_fn, *args, std_trick=True, residual_trick=True)

    def reverse(self, score_fn):
        return RSDE(self, score_fn)

    def probability_ode(self, score_fn):
        return ProbabilityFlowODE(self, score_fn)


class ProbabilityFlowODE:
    def __init__(self, sde: SDE, score_fn=None):
        self.sde = sde

        self.t0 = sde.t0
        self.tf = sde.tf

        if score_fn is None and not isinstance(sde, RSDE):
            raise ValueError(
                "Score function must be not None or SDE must be a reversed SDE"
            )
        elif score_fn is not None:
            self.score_fn = score_fn
        elif isinstance(sde, RSDE):
            self.score_fn = sde.score_fn

    def coefficients(self, x, t, z=None):
        drift, diffusion = self.sde.coefficients(x, t)
        score_fn = self.score_fn(x, t, z)
        # compute G G^T score_fn
        if self.sde.full_diffusion_matrix:
            # if square matrix diffusion coeffs
            ode_drift = drift - 0.5 * jnp.einsum(
                "...ij,...kj,...k->...i", diffusion, diffusion, score_fn
            )
        else:
            # if scalar diffusion coeffs (i.e. no extra dims on the diffusion)
            ode_drift = drift - 0.5 * jnp.einsum(
                "...,...,...i->...i", diffusion, diffusion, score_fn
            )

        return ode_drift, jnp.zeros(drift.shape[:-1])


def get_matrix_div_fn(func):
    def matrix_div_fn(x, t, context):
        # define function that returns div of nth column matrix function
        f = lambda n: get_exact_div_fn(lambda x, t, context: func(x, t, context)[..., n])(
            x, t, context
        )
        matrix = func(x, t, context)
        div_term = jax.vmap(f)(jnp.arange(matrix.shape[-1]))
        div_term = jnp.moveaxis(div_term, 0, -1)
        return div_term

    return matrix_div_fn


class RSDE(SDE):
    """Reverse time SDE, assuming the diffusion coefficient is spatially homogenous"""

    def __init__(self, sde: SDE, score_fn):
        super().__init__(sde.beta_schedule.reverse())
        self.sde = sde
        self.score_fn = score_fn

    def diffusion(self, x, t):
        return self.sde.diffusion(x, t)

    def drift(self, x, t):
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

        if self.sde.spatial_dependent_diffusion:
            # NOTE: this has not been tested
            if self.sde.full_diffusion_matrix:
                # ∇·(G G^t) = (∇· G_i G_i^t)_i =
                G_G_tr = lambda x, t, _: jnp.einsum(
                    "...ij,...kj->...ik",
                    self.sde.diffusion(x, t),
                    self.sde.diffusion(x, t),
                )
                matrix_div_fn = get_matrix_div_fn(G_G_tr)
                div_term = matrix_div_fn(x, t, None)
            else:
                # ∇·(g^2 I) = (∇·g^2 1_d)_i = (||∇ g^2||_1)_i = 2 g ||∇ g||_1 1_d
                grad = jax.vmap(jax.grad(self.sde.diffusion, argnums=0))(x, t)
                ones = jnp.ones_like(forward_drift)
                div_term = 2 * diffusion[..., None] * grad.sum(axis=-1)[..., None] * ones
            reverse_drift += div_term

        return reverse_drift

    def reverse(self):
        return self.sde


class VPSDE(SDE):
    def __init__(self, beta_schedule):
        from score_sde.models.distribution import NormalDistribution

        super().__init__(beta_schedule)
        self.limiting = NormalDistribution()

    def drift(self, x, t):
        beta_t = self.beta_schedule.beta_t(t)
        return -0.5 * beta_t[..., None] * x

    def marginal_prob(self, x, t):
        log_mean_coeff = self.beta_schedule.log_mean_coeff(t)
        mean = jnp.exp(log_mean_coeff)[..., None] * x
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std


class VESDE(SDE):
    pass
