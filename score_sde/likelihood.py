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
from typing import Sequence
import jax
import numpy as np
import jax.numpy as jnp
from scipy import integrate

from score_sde.sde import SDE
from score_sde.utils import (
    ParametrisedScoreFunction,
    TrainState,
    get_div_fn,
    get_estimate_div_fn,
)
from score_sde.utils import (
    get_score_fn,
    to_flattened_numpy,
    from_flattened_numpy,
    unreplicate,
)


@jax.pmap
def p_div_fn(
    train_state: TrainState, hutchinson_type: str, x: jnp.ndarray, t: float, eps: jnp.ndarray
) -> jnp.ndarray:
    """Pmapped divergence of the drift function."""
    if hutchinson_type == "None":
        return get_div_fn(lambda x, t: drift_fn(train_state, x, t))(x, t)
    else:
        return get_estimate_div_fn(lambda x, t: drift_fn(train_state, x, t))(
            x, t, eps
        )


def drift_fn(
    sde: SDE,
    model: ParametrisedScoreFunction,
    train_state: TrainState,
    x: jnp.ndarray,
    t: float) -> jnp.ndarray:
    """The drift function of the reverse-time SDE."""
    score_fn = get_score_fn(
        sde,
        model,
        train_state.params_ema,
        train_state.model_state,
        train=False,
        continuous=True,
    )
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]


p_drift_fn = jax.pmap(drift_fn)  # Pmapped drift function of the reverse-time SDE


def div_noise(rng: jax.random.KeyArray, shape: Sequence[int], hutchinson_type: str) -> jnp.ndarray:
    """Sample noise for the hutchinson estimator."""
    if hutchinson_type == "Gaussian":
        epsilon = jax.random.normal(rng, shape)
    elif hutchinson_type == "Rademacher":
        epsilon = (
            jax.random.randint(rng, shape, minval=0, maxval=2).astype(
                jnp.float32
            )
            * 2
            - 1
        )
    elif hutchinson_type == "None":
        epsilon = None
    else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
    return epsilon


def get_likelihood_fn(
    sde: SDE,
    model: ParametrisedScoreFunction,
    inverse_scaler,
    hutchinson_type: str = "Rademacher",
    rtol: str = 1e-5,
    atol: str = 1e-5,
    method: str = "RK45",
    eps: str = 1e-5,
    bits_per_dimension=True,
):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      inverse_scaler: The inverse data normalizer.
      hutchinson_type: "Rademacher", "Gaussian" or "None". The type of noise for Hutchinson-Skilling trace estimator.
      rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
      atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
      method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
      eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states, replicated training states, and a batch of data points
        and returns the log-likelihoods in bits/dim, the latent code, and the number of function
        evaluations cost by computation.
    """
    p_prior_logp_fn = jax.pmap(
        sde.prior_logp
    )  # Pmapped log-PDF of the SDE's prior distribution

    def likelihood_fn(
        prng: jax.random.KeyArray, ptrain_state: TrainState, data: jnp.ndarray
    ):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
          prng: An array of random states. The list dimension equals the number of devices.
          ptrain_state: Replicated training state for running on multiple devices.
          data: A JAX array of shape [#devices, batch size, ...].

        Returns:
          bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
          z: A JAX array of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
          nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        rng, step_rng = jax.random.split(unreplicate(prng))
        shape = data.shape
        epsilon = div_noise(step_rng, shape, hutchinson_type)

        def ode_func(t: float, x: jnp.ndarray) -> np.array:
            sample = from_flattened_numpy(x[: -shape[0] * shape[1]], shape)
            vec_t = jnp.ones((sample.shape[0], sample.shape[1])) * t
            drift = to_flattened_numpy(p_drift_fn(sde, model, ptrain_state, sample, vec_t))
            logp_grad = to_flattened_numpy(
                p_div_fn(ptrain_state, hutchinson_type, sample, vec_t, epsilon)
            )
            return np.concatenate([drift, logp_grad], axis=0)

        init = jnp.concatenate(
            [to_flattened_numpy(data), np.zeros((shape[0] * shape[1],))], axis=0
        )
        solution = integrate.solve_ivp(
            ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method
        )
        nfe = solution.nfev
        zp = jnp.asarray(solution.y[:, -1])
        z = from_flattened_numpy(zp[: -shape[0] * shape[1]], shape)
        delta_logp = zp[-shape[0] * shape[1] :].reshape((shape[0], shape[1]))
        prior_logp = p_prior_logp_fn(z)
        posterior_logp = prior_logp + delta_logp
        bpd = -posterior_logp / np.log(2)
        N = np.prod(shape[2:])
        bpd = bpd / N
        # A hack to convert log-likelihoods to bits/dim
        # based on the gradient of the inverse data normalizer.
        offset = jnp.log2(jax.grad(inverse_scaler)(0.0)) + 8.0
        bpd += offset
        return bpd if bits_per_dimension else posterior_logp, z, nfe

    return likelihood_fn
