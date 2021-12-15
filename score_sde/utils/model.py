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

"""All functions and modules related to model definition.
"""
import functools
from dataclasses import dataclass
from typing import Any, NamedTuple, Callable

import jax
import haiku as hk
import optax
import numpy as np
import jax.numpy as jnp

from .jax import batch_mul
from .registry import register_category
from .typing import ParametrisedScoreFunction

from score_sde.sde import SDE, VESDE, VPSDE, subVPSDE, Brownian


def get_sigmas(sigma_min, sigma_max, num_scales):
    """Get sigmas --- the set of noise levels for SMLD."""
    sigmas = jnp.exp(
        jnp.linspace(
            jnp.log(sigma_max),
            jnp.log(sigma_min),
            num_scales,
        )
    )

    return sigmas


def get_score_fn(
    sde: SDE,
    model: ParametrisedScoreFunction,
    params,
    state,
    train=False,
    continuous=False,
    return_state=False,
):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde.SDE` object that represents the forward SDE.
      model: A Haiku transformed function representing the score function model
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all other mutable parameters.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.
      return_state: If `True`, return the new mutable states alongside the model output.

    Returns:
      A score function.
    """
    # model_fn = get_model_fn(model, params, states, train=train)

    if isinstance(sde, (VPSDE, subVPSDE, Brownian)):

        def score_fn(x, t, rng=None):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                model_out, new_state = model.apply(params, state, rng, x=x, t=labels)
                std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                model_out, new_state = model.apply(params, state, rng, x=x, t=labels)
                std = sde.sqrt_1m_alphas_cumprod[labels.astype(jnp.int32)]

            score = batch_mul(-model_out, 1.0 / std)
            if return_state:
                return score, new_state
            else:
                return score

    elif isinstance(sde, VESDE):

        def score_fn(x, t, rng=None):
            if continuous:
                labels = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = jnp.round(labels).astype(jnp.int32)

            score, state = model(x, labels, rng)
            if return_state:
                return score, state
            else:
                return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn


def get_model_fn(model, params, state, train=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: A transformed Haiku function the represent the architecture of score-based model.
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all mutable states.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels, rng=None):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
          rng: If present, it is the random state for dropout

        Returns:
          A tuple of (model output, new mutable states)
        """
        return model.apply(
            parmas, state, rng if rng is not None else hk.next_rng_key(), x, lables
        )

    return model_fn


@dataclass
class ScoreFunctionWrapper:
    model: object()

    def __call__(self, x, t):
        t = jnp.array(t)
        if len(t.shape) == 0:
            t = t * jnp.ones(x.shape[:-1])

        if len(t.shape) == len(x.shape) - 1:
            t = jnp.expand_dims(t, axis=-1)

        return self.model(jnp.concatenate([x, t], axis=-1))
