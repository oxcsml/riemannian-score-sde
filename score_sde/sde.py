"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
Modified code from https://github.com/yang-song/score_sde
"""
import abc
from typing import Tuple, Callable

import jax
import numpy as np
import jax.numpy as jnp
from geomstats.geometry.euclidean import Euclidean

# from .utils import batch_mul
def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, T: float, N: int):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.T = T
        self.N = N

    @abc.abstractmethod
    def sde(self, x: jnp.ndarray, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the drift and diffusion coefficients for the SDE"""
        raise NotImplementedError()

    @abc.abstractmethod
    def marginal_prob(
        self, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Parameters (mean, std) to determine the marginal distribution of the SDE, $p_t(x)$."""
        raise NotImplementedError()

    @abc.abstractmethod
    def prior_sampling(
        self, rng: jax.random.KeyArray, shape: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Generate sample from the prior distribution, $p_T(x)$.
        Shape gives the expected shape of a single sample
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def prior_logp(self, z: jnp.ndarray) -> jnp.ndarray:
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        raise NotImplementedError()

    def discretize(self, x: jnp.ndarray, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization. TODO: Note sure what they mean by this?

        Args:
          x: a JAX tensor.
          t: a JAX float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * jnp.sqrt(dt)
        return f, G

    def reverse(
        self,
        score_fn: Callable[[jnp.ndarray], jnp.ndarray],
        probability_flow: bool = False,
    ):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - batch_mul(
                    diffusion ** 2, score * (0.5 if self.probability_flow else 1.0)
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = (
                    jnp.zeros_like(diffusion) if self.probability_flow else diffusion
                )
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - batch_mul(
                    G ** 2, score_fn(x, t) * (0.5 if self.probability_flow else 1.0)
                )
                rev_G = jnp.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, manifold, T, beta_0, beta_1, N):
        """Construct a Variance Preserving SDE.

        Args:
          beta_0: value of beta(0)
          beta_1: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(T, N)
        assert isinstance(manifold, Euclidean)
        self.manifold = manifold
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.discrete_betas = jnp.linspace(beta_0 / N, beta_1 / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * batch_mul(beta_t, x)
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, rng, shape):
        return jax.random.normal(rng, shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logp_fn = lambda z: -N / 2.0 * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.0
        return jax.vmap(logp_fn)(z)

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).astype(jnp.int32)
        beta = self.discrete_betas[timestep]
        alpha = self.alphas[timestep]
        sqrt_beta = jnp.sqrt(beta)
        f = batch_mul(jnp.sqrt(alpha), x) - x
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * batch_mul(beta_t, x)
        discount = 1.0 - jnp.exp(
            -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2
        )
        diffusion = jnp.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = 1 - jnp.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, rng, shape):
        return jax.random.normal(rng, shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logp_fn = lambda z: -N / 2.0 * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.0
        return jax.vmap(logp_fn)(z)


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = jnp.exp(
            np.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        )
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = jnp.zeros_like(x)
        diffusion = sigma * jnp.sqrt(
            2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min))
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, rng, shape):
        return jax.random.normal(rng, shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logp_fn = lambda z: -N / 2.0 * jnp.log(
            2 * np.pi * self.sigma_max ** 2
        ) - jnp.sum(z ** 2) / (2 * self.sigma_max ** 2)
        return jax.vmap(logp_fn)(z)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).astype(jnp.int32)
        sigma = self.discrete_sigmas[timestep]
        adjacent_sigma = jnp.where(
            timestep == 0, jnp.zeros_like(timestep), self.discrete_sigmas[timestep - 1]
        )
        f = jnp.zeros_like(x)
        G = jnp.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G


class Brownian(SDE):

    def __init__(self, manifold, T, beta_0, beta_1, N):
        """Construct a Brownian motion on a compact manifold.

        Args:
          N: number of discretization steps
        """
        super().__init__(T, N)
        self.manifold = manifold
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        # beta_t = jnp.ones_like(x)
        drift = jnp.zeros_like(x)
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """ Should not rely on closed-form marginal probability """
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        # mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        # return mean, std
        return jnp.zeros_like(x), std

    def marginal_sample(self, rng, x, t):
        from score_sde.sampling import EulerMaruyamaManifoldPredictor, get_pc_sampler  # TODO: remove from class

        perturbed_x = self.manifold.random_walk(rng, x, t)
        if perturbed_x is None:
            # TODO: should pmap the pc_sampler?
            sampler = get_pc_sampler(self, None, x.shape, predictor=EulerMaruyamaManifoldPredictor, corrector=None, continuous=True, forward=True)
            perturbed_x, _ = sampler(rng, None, x, t)
        return perturbed_x

    def marginal_log_prob(self, x0, x, t, **kwargs):
        # TODO: Should indeed vmap?
        # NOTE: reshape: https://github.com/google/jax/issues/2303
        s = 2 * (0.25 * t ** 2 * (self.beta_1 - self.beta_0) + 0.5 * t * self.beta_0)
        return jnp.reshape(self.manifold.log_heat_kernel(x0, x, s, **kwargs), ())

    def grad_marginal_log_prob(self, x0, x, t):
        logp_grad_fn = jax.value_and_grad(self.marginal_log_prob, argnums=1, has_aux=False)
        logp, logp_grad = jax.vmap(logp_grad_fn)(x0, x, t)
        logp_grad = self.manifold.to_tangent(logp_grad, x)
        return logp, logp_grad

    def prior_sampling(self, rng, shape):
        return self.manifold.random_uniform(state=rng, n_samples=shape[0])

    def prior_logp(self, z):
        return - jnp.ones([*z.shape[:-1]]) * self.manifold.metric.log_volume
