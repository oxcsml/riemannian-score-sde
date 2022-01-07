"""Modified code from https://github.com/yang-song/score_sde"""
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import abc
import functools
from scipy import integrate
from typing import Callable, Tuple

import jax
import haiku
import jax.numpy as jnp
import jax.random as random

from score_sde.sde import SDE, VPSDE, subVPSDE, VESDE
from score_sde.models import get_score_fn
from score_sde.utils import from_flattened_numpy, to_flattened_numpy
from score_sde.utils import batch_mul, batch_add
from score_sde.utils import register_category
from score_sde.utils import ParametrisedScoreFunction, ScoreFunction, SDEUpdateFunction
from score_sde.utils import TrainState

get_predictor, register_predictor = register_category("predictors")
get_corrector, register_corrector = register_category("correctors")


# def get_sampling_fn(
#     sampling_method: str,
#     sde: sde.SDE,
#     model: flax.l,
#     shape,
#     inverse_scaler,
#     eps,
#     noise_removal: bool,
# ):
#     """Create a sampling function.

#     Args:
#       config: A `ml_collections.ConfigDict` object that contains all configuration information.
#       sde: A `sde_lib.SDE` object that represents the forward SDE.
#       model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
#       shape: A sequence of integers representing the expected shape of a single sample.
#       inverse_scaler: The inverse data normalizer function.
#       eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

#     Returns:
#       A function that takes random states and a replicated training state and outputs samples with the
#         trailing dimensions matching `shape`.
#     """

#     sampler_name = sampling.method
#     # Probability flow ODE sampling with black-box ODE solvers
#     if sampler_name.lower() == "ode":
#         sampling_fn = get_ode_sampler(
#             sde=sde,
#             model=model,
#             shape=shape,
#             inverse_scaler=inverse_scaler,
#             denoise=noise_removal,
#             eps=eps,
#         )
#     # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
#     elif sampler_name.lower() == "pc":
#         predictor = get_predictor(config.sampling.predictor.lower())
#         corrector = get_corrector(config.sampling.corrector.lower())
#         sampling_fn = get_pc_sampler(
#             sde=sde,
#             model=model,
#             shape=shape,
#             predictor=predictor,
#             corrector=corrector,
#             inverse_scaler=inverse_scaler,
#             snr=config.sampling.snr,
#             n_steps=config.sampling.n_steps_each,
#             probability_flow=config.sampling.probability_flow,
#             continuous=config.training.continuous,
#             denoise=config.sampling.noise_removal,
#             eps=eps,
#         )
#     else:
#         raise ValueError(f"Sampler name {sampler_name} unknown.")

#     return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(
        self,
        sde: SDE,
        score_fn: ParametrisedScoreFunction,
        forward: bool,
        probability_flow=False,
    ):
        super().__init__()
        self.sde = sde
        self.forward = forward
        # Compute the reverse SDE/ODE
        self.rsde = sde if forward else sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """One update of the predictor.

        Args:
          rng: A JAX random state.
          x: A JAX array representing the current state
          t: A JAX array representing the current time step.

        Returns:
          x: A JAX array of the next state.
          x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        raise NotImplementedError()


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(
        self, sde: SDE, score_fn: ParametrisedScoreFunction, forward: bool, snr: float, n_steps: int
    ):
        super().__init__()
        self.sde = sde
        self.forward = forward
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """One update of the corrector.

        Args:
          rng: A JAX random state.
          x: A JAX array representing the current state
          t: A JAX array representing the current time step.

        Returns:
          x: A JAX array of the next state.
          x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        raise NotImplementedError()


@register_predictor
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, forward, probability_flow=False):
        super().__init__(sde, score_fn, forward, probability_flow)

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        dt = self.rsde.T / self.rsde.N
        sign = 1 if self.forward else -1
        z = random.normal(rng, x.shape)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + sign * drift * dt
        x = x_mean + batch_mul(diffusion, jnp.sqrt(dt) * z)
        return x, x_mean


@register_predictor
class EulerMaruyamaManifoldPredictor(Predictor):
    def __init__(self, sde, score_fn, forward, probability_flow=False):
        super().__init__(sde, score_fn, forward, probability_flow)
        self.forward = forward

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        dt = self.rsde.T / self.rsde.N
        sign = 1 if self.forward else -1
        rng, z = self.sde.manifold.random_normal_tangent(state=rng, base_point=x, n_samples=x.shape[0])
        drift, diffusion = self.rsde.sde(x, t)
        drift = sign * drift * dt
        #NOTE: should we use retraction?
        x_mean = self.sde.manifold.metric.exp(tangent_vec=drift, base_point=x)  # NOTE: do we really need this in practice? only if denoise=True
        tangent_vector = drift + batch_mul(diffusion, jnp.sqrt(dt) * z)
        x = self.sde.manifold.metric.exp(tangent_vec=tangent_vector, base_point=x)
        return x, x_mean


@register_predictor
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, forward, probability_flow=False):
        super().__init__(sde, score_fn, forward, probability_flow)

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        f, G = self.rsde.discretize(x, t)
        z = random.normal(rng, x.shape)
        x_mean = x - f
        x = x_mean + batch_mul(G, z)
        return x, x_mean


@register_predictor
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, forward, probability_flow=False):
        super().__init__(sde, score_fn, forward, probability_flow)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        assert (
            not probability_flow
        ), "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = jnp.where(
            timestep == 0, jnp.zeros(t.shape), sde.discrete_sigmas[timestep - 1]
        )
        score = self.score_fn(x, t)
        x_mean = x + batch_mul(score, sigma ** 2 - adjacent_sigma ** 2)
        std = jnp.sqrt(
            (adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2)
        )
        noise = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, noise)
        return x, x_mean

    def vpsde_update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
        beta = sde.discrete_betas[timestep]
        score = self.score_fn(x, t)
        x_mean = batch_mul((x + batch_mul(beta, score)), 1.0 / jnp.sqrt(1.0 - beta))
        noise = random.normal(rng, x.shape)
        x = x_mean + batch_mul(jnp.sqrt(beta), noise)
        return x, x_mean

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if isinstance(self.sde, VESDE):
            return self.vesde_update_fn(rng, x, t)
        elif isinstance(self.sde, VPSDE):
            return self.vpsde_update_fn(rng, x, t)


@register_predictor
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, forward, probability_flow=False):
        pass

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return x, x


@register_corrector
class LangevinCorrector(Corrector):
    def __init__(
        self, sde: SDE, score_fn: ParametrisedScoreFunction, forward: bool, snr: float, n_steps: int
    ):
        super().__init__(sde, score_fn, forward, snr, n_steps)
        if (
            not isinstance(sde, VPSDE)
            and not isinstance(sde, VESDE)
            and not isinstance(sde, subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
            alpha = sde.alphas[timestep]
        else:
            alpha = jnp.ones_like(t)

        def loop_body(step, val):
            rng, x, x_mean = val
            grad = score_fn(x, t)
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, x.shape)
            grad_norm = jnp.linalg.norm(
                grad.reshape((grad.shape[0], -1)), axis=-1
            ).mean()
            grad_norm = jax.lax.pmean(grad_norm, axis_name="batch")
            noise_norm = jnp.linalg.norm(
                noise.reshape((noise.shape[0], -1)), axis=-1
            ).mean()
            noise_norm = jax.lax.pmean(noise_norm, axis_name="batch")
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + batch_mul(step_size, grad)
            x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
        return x, x_mean


@register_corrector
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    We include this corrector only for completeness. It was not directly used in our paper.
    """

    def __init__(
        self, sde: SDE, score_fn: ParametrisedScoreFunction, forward: bool, snr: float, n_steps: int
    ):
        super().__init__(sde, score_fn, forward, snr, n_steps)
        if (
            not isinstance(sde, VPSDE)
            and not isinstance(sde, VESDE)
            and not isinstance(sde, subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
            alpha = sde.alphas[timestep]
        else:
            alpha = jnp.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        def loop_body(step, val):
            rng, x, x_mean = val
            grad = score_fn(x, t)
            rng, step_rng = jax.random.split(rng)
            noise = jax.random.normal(step_rng, x.shape)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + batch_mul(step_size, grad)
            x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
        return x, x_mean


@register_corrector
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(
        self, sde: SDE, score_fn: ParametrisedScoreFunction, forward: bool, snr: float, n_steps: int
    ):
        pass

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return x, x


def shared_predictor_update_fn(
    rng: jax.random.KeyArray,
    train_state: TrainState,
    x: jnp.ndarray,
    t: jnp.ndarray,
    sde: SDE,
    model: ParametrisedScoreFunction,
    predictor: Predictor,
    forward: bool,
    probability_flow: bool,
    continuous: bool,
) -> SDEUpdateFunction:
    """A wrapper that configures and returns the update function of predictors."""
    if forward:
        score_fn = None
    else:
        score_fn = get_score_fn(
            sde,
            model,
            train_state.params_ema,
            train_state.model_state,
            train=False,
            continuous=continuous,
        )
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, forward, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, forward, probability_flow)
    return predictor_obj.update_fn(rng, x, t)


def shared_corrector_update_fn(
    rng: jax.random.KeyArray,
    train_state: TrainState,
    x: jnp.ndarray,
    t: jnp.ndarray,
    sde: SDE,
    model: ParametrisedScoreFunction,
    corrector: Corrector,
    forward: bool,
    continuous: bool,
    snr: float,
    n_steps: int,
) -> SDEUpdateFunction:
    """A wrapper tha configures and returns the update function of correctors."""
    if forward:
        score_fn = None
    else:
        score_fn = get_score_fn(
            sde,
            model,
            train_state.params_ema,
            train_state.model_state,
            train=False,
            continuous=continuous,
        )
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, forward, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, forward, snr, n_steps)
    return corrector_obj.update_fn(rng, x, t)


def get_pc_sampler(
    sde: SDE,
    model: ParametrisedScoreFunction,
    shape,
    predictor: Predictor,
    corrector: Corrector,
    forward: bool = False,
    inverse_scaler=lambda x: x,  # TODO: Figure type
    snr: float = 0.2,
    n_steps: int = 1,
    probability_flow: bool = False,
    continuous: bool = False,
    denoise: bool = True,
    eps: float = 1e-3,
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.

    Returns:
      A sampling function that takes random states, and a replcated training state and returns samples as well as
      the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        model=model,
        predictor=predictor,
        forward=forward,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        model=model,
        corrector=corrector,
        forward=forward,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_sampler(rng, state=None, x=None, T=None):
        """The PC sampler funciton.

        Args:
          rng: A JAX random state
          state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
        Returns:
          Samples, number of function evaluations
        """
        # Initial sample
        rng, step_rng = random.split(rng)
        if forward:
            assert (x is not None) and (T is not None)
        else:
            x = sde.prior_sampling(step_rng, shape)
            T = sde.T

        N = int(jnp.round(T / sde.T * sde.N))
        timesteps = jnp.linspace(T, eps, N)
        timesteps = jnp.flip(timesteps) if forward else timesteps

        def loop_body(i, val):
            rng, x, x_mean = val
            t = timesteps[i]
            vec_t = jnp.ones(shape[0]) * t
            rng, step_rng = random.split(rng)
            x, x_mean = corrector_update_fn(step_rng, state, x, vec_t)
            rng, step_rng = random.split(rng)
            x, x_mean = predictor_update_fn(step_rng, state, x, vec_t)
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, N, loop_body, (rng, x, x))
        # Denoising is equivalent to running one predictor step without adding noise.
        return inverse_scaler(x_mean if denoise else x), N * (n_steps + 1)

    return pc_sampler
    # return jax.pmap(pc_sampler, axis_name="batch")


def get_ode_sampler(
    sde: SDE,
    model_fn: ParametrisedScoreFunction,
    shape,
    inverse_scaler,
    denoise: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    method: str = "RK45",
    eps: float = 1e-3,
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      shape: A sequence of integers. The expected shape of a single sample.
      inverse_scaler: The inverse data normalizer.
      denoise: If `True`, add one-step denoising to final samples.
      rtol: A `float` number. The relative tolerance level of the ODE solver.
      atol: A `float` number. The absolute tolerance level of the ODE solver.
      method: A `str`. The algorithm used for the black-box ODE solver.
        See the documentation of `scipy.integrate.solve_ivp`.
      eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.

    Returns:
      A sampling function that takes random states, and a replicated training state and returns samples
      as well as the number of function evaluations during sampling.
    """

    @jax.pmap
    def denoise_update_fn(rng, state, x):
        score_fn = get_score_fn(
            sde,
            model_fn,
            state.params_ema,
            state.model_state,
            train=False,
            continuous=True,
        )
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = jnp.ones((x.shape[0],)) * eps
        _, x = predictor_obj.update_fn(rng, x, vec_eps)
        return x

    @jax.pmap
    def drift_fn(state, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(
            sde,
            model_fn,
            state.params_ema,
            state.model_state,
            train=False,
            continuous=True,
        )
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(prng, pstate, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
          prng: An array of random state. The leading dimension equals the number of devices.
          pstate: Replicated training state for running on multiple devices.
          z: If present, generate samples from latent code `z`.
        Returns:
          Samples, and the number of function evaluations.
        """
        # Initial sample
        rng = flax.jax_utils.unreplicate(prng)
        rng, step_rng = random.split(rng)
        if z is None:
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = sde.prior_sampling(step_rng, (jax.local_device_count(),) + shape)
        else:
            x = z

        def ode_func(t, x):
            x = from_flattened_numpy(x, (jax.local_device_count(),) + shape)
            vec_t = jnp.ones((x.shape[0], x.shape[1])) * t
            drift = drift_fn(pstate, x, vec_t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(
            ode_func,
            (sde.T, eps),
            to_flattened_numpy(x),
            rtol=rtol,
            atol=atol,
            method=method,
        )
        nfe = solution.nfev
        x = jnp.asarray(solution.y[:, -1]).reshape((jax.local_device_count(),) + shape)

        # Denoising is equivalent to running one predictor step without adding noise
        if denoise:
            rng, *step_rng = random.split(rng, jax.local_device_count() + 1)
            step_rng = jnp.asarray(step_rng)
            x = denoise_update_fn(step_rng, pstate, x)

        x = inverse_scaler(x)
        return x, nfe

    return ode_sampler
