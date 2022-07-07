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
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as random

from score_sde.sde import SDE, RSDE

from score_sde.utils import register_category

get_predictor, register_predictor = register_category("predictors")
get_corrector, register_corrector = register_category("correctors")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(
        self,
        sde: SDE,
    ):
        super().__init__()
        self.sde = sde

    @abc.abstractmethod
    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float, dt: float
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
        self,
        sde: SDE,
        snr: float,
        n_steps: int,
    ):
        super().__init__()
        self.sde = sde
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float, dt: float
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
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        z = jax.random.normal(rng, x.shape)
        drift, diffusion = self.sde.coefficients(x, t)
        x_mean = x + drift * dt[..., None]

        if len(diffusion.shape) > 1 and diffusion.shape[-1] == diffusion.shape[-2]:
            # if square matrix diffusion coeffs
            diffusion_term = jnp.einsum(
                "...ij,j,...->...i", diffusion, z, jnp.sqrt(jnp.abs(dt))
            )
        else:
            # if scalar diffusion coeffs (i.e. no extra dims on the diffusion)
            diffusion_term = jnp.einsum(
                "...,...i,...->...i", diffusion, z, jnp.sqrt(jnp.abs(dt))
            )

        x = x_mean + diffusion_term
        return x, x_mean


@register_predictor
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde):
        pass

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return x, x


@register_corrector
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(
        self,
        sde: SDE,
        snr: float,
        n_steps: int,
    ):
        pass

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return x, x


def get_pc_sampler(
    sde: SDE,
    N: int,
    predictor: Predictor = "EulerMaruyamaPredictor",
    corrector: Corrector = None,
    inverse_scaler=lambda x: x,
    snr: float = 0.2,
    n_steps: int = 1,
    denoise: bool = True,
    eps: float = 1e-3,
    return_hist=False,
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
      N: An `integer` specifying the number of steps to perform in the sampler

    Returns:
      A sampling function that takes random states, and a replcated training state and returns samples as well as
      the number of function evaluations during sampling.
    """
    predictor = get_predictor(predictor if predictor is not None else "NonePredictor")(
        sde
    )
    corrector = get_corrector(corrector if corrector is not None else "NoneCorrector")(
        sde, snr, n_steps
    )

    def pc_sampler(rng, x, t0=None, tf=None):
        """The PC sampler funciton.

        Args:
          rng: A JAX random state
          state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
        Returns:
          Samples, number of function evaluations
        """

        t0 = sde.t0 if t0 is None else t0
        tf = sde.tf if tf is None else tf
        t0 = jnp.broadcast_to(t0, x.shape[0])
        tf = jnp.broadcast_to(tf, x.shape[0])

        # Only integrate to eps off the forward start time for numerical stability
        if isinstance(sde, RSDE):
            tf = tf + eps
        else:
            t0 = t0 + eps

        timesteps = jnp.linspace(start=t0, stop=tf, num=N, endpoint=True)
        dt = (tf - t0) / N

        def loop_body(i, val):
            rng, x, x_mean, x_hist = val
            t = timesteps[i]
            rng, step_rng = random.split(rng)
            x, x_mean = corrector.update_fn(step_rng, x, t, dt)
            rng, step_rng = random.split(rng)
            x, x_mean = predictor.update_fn(step_rng, x, t, dt)

            x_hist = x_hist.at[i].set(x)

            return rng, x, x_mean, x_hist

        x_hist = jnp.zeros((N, *x.shape))

        _, x, x_mean, x_hist = jax.lax.fori_loop(0, N, loop_body, (rng, x, x, x_hist))
        # Denoising is equivalent to running one predictor step without adding noise.
        if return_hist:
            return (
                inverse_scaler(x_mean if denoise else x),
                inverse_scaler(x_hist),
                timesteps,
            )
        else:
            return inverse_scaler(x_mean if denoise else x)

    return pc_sampler
