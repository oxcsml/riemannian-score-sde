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
from dataclasses import dataclass

import jax
import numpy as np
import jax.numpy as jnp

from score_sde.utils.jax import batch_mul
from score_sde.utils.typing import ParametrisedScoreFunction

from score_sde.sde import SDE, VESDE, VPSDE, subVPSDE
from riemannian_score_sde.sde import Brownian


def get_score_fn(
    sde: SDE,
    model: ParametrisedScoreFunction,
    params,
    state,
    train=False,
    return_state=False,
):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
    Args:
      sde: An `sde.SDE` object that represents the forward SDE.
      model: A Haiku transformed function representing the score function model
      params: A dictionary that contains all trainable parameters.
      states: A dictionary that contains all other mutable parameters.
      train: `True` for training and `False` for evaluation.
      return_state: If `True`, return the new mutable states alongside the model output.
    Returns:
      A score function.
    """
    if isinstance(sde, Brownian):
        def score_fn(x, t, context=None, std_trick=True, rng=None):
            if context is not None:
                t_expanded = jnp.expand_dims(t.reshape(-1), -1)
                if context.shape[0] != x.shape[0]:
                    context = jnp.repeat(jnp.expand_dims(context, 0), x.shape[0], 0)
                context = jnp.concatenate([t_expanded, context], axis=-1)
            else:
                context = t
            model_out, new_state = model.apply(params, state, rng, x=x, t=context)
            # NOTE: scaling the output with 1.0 / std helps cf 'Improved Techniques for Training Score-Based Generative Model'
            score = model_out
            if std_trick:
                std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
                score = batch_mul(model_out, 1.0 / std)
            if return_state:
                return score, new_state
            else:
                return score

    elif isinstance(sde, (VPSDE, subVPSDE)):
        def score_fn(x, t, context=None, rng=None):
            # Scale neural network output by standard deviation and flip sign
            # For VP-trained models, t=0 corresponds to the lowest noise level
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.
            if context is not None:
                t_expanded = jnp.expand_dims((t * 999).reshape(-1), -1)
                context = jnp.concatenate([t_expanded, context], axis=-1)
            else:
                context = t * 999  # TODO: remove?
            model_out, new_state = model.apply(params, state, rng, x=x, t=context)
            std = sde.marginal_prob(jnp.zeros_like(x), t)[1]

            score = batch_mul(-model_out, 1.0 / std)
            if return_state:
                return score, new_state
            else:
                return score

    # elif isinstance(sde, VESDE):

    #     def score_fn(x, t, rng=None):
    #         labels = sde.marginal_prob(jnp.zeros_like(x), t)[1]
    #         score, state = model(x, labels, rng)
    #         if return_state:
    #             return score, state
    #         else:
    #             return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn
