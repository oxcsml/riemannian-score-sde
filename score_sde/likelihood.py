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
import jax
import numpy as np
import jax.numpy as jnp

from score_sde.sde import SDE, ProbabilityFlowODE
from score_sde.utils import ParametrisedScoreFunction
from score_sde.ode import odeint
from score_sde.models import get_div_fn, div_noise

def get_likelihood_fn(
    sde: SDE,
    score_fn: ParametrisedScoreFunction,
    inverse_scaler=lambda x: x,
    hutchinson_type: str = "Rademacher",
    rtol: str = 1e-5,
    atol: str = 1e-5,
    method: str = "RK45",
    eps: str = 1e-5,
    bits_per_dimension=True,
):
    def likelihood_fn(rng: jax.random.KeyArray, data: jnp.ndarray, tf : float = None):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
          rng: An array of random states. The list dimension equals the number of devices.
          train_state: Replicated training state for running on multiple devices.
          data: A JAX array of shape [#devices, batch size, ...].

        Returns:
          bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
          z: A JAX array of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
          nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        pode = ProbabilityFlowODE(sde, score_fn)
        drift_fn = lambda x, t: pode.coefficients(x, t)[0]
        div_fn = get_div_fn(drift_fn, hutchinson_type)
        drift_fn, div_fn = jax.jit(drift_fn), jax.jit(div_fn)

        rng, step_rng = jax.random.split(rng)
        shape = data.shape
        epsilon = div_noise(step_rng, shape, hutchinson_type)
        tf = sde.tf if tf is None else tf

        def ode_func(x: jnp.ndarray, t: jnp.ndarray) -> np.array:
            sample = x[:, :shape[1]]
            vec_t = jnp.ones((sample.shape[0],)) * t
            drift = drift_fn(sample, vec_t)
            logp_grad = div_fn(sample, vec_t, epsilon).reshape([*shape[:-1], 1])
            return jnp.concatenate([drift, logp_grad], axis=1)

        init = jnp.concatenate([data, np.zeros((shape[0], 1))], axis=1)
        ts = jnp.array([eps, tf])
        y, nfe = odeint(ode_func, init, ts, rtol=rtol, atol=atol)

        z = y[-1, ..., :-1]
        delta_logp = y[-1, ..., -1]

        prior_logp = sde.limiting_distribution_logp(z)
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