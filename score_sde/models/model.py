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
import jax
import numpy as np
import jax.numpy as jnp

from score_sde.utils.jax import batch_mul
from score_sde.utils.typing import ParametrisedScoreFunction


def get_score_fn(
    sde,
    model: ParametrisedScoreFunction,
    params,
    state,
    train=False,
    return_state=False,
    std_trick=True,
    residual_trick=True,
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

    def score_fn(y, t, context, rng=None):
        model_out, new_state = model.apply(params, state, rng, y=y, t=t, context=context)
        score = model_out

        if std_trick:
            # NOTE: scaling the output with 1.0 / std helps cf 'Improved Techniques for Training Score-Based Generative Model'
            std = sde.marginal_prob(jnp.zeros_like(y), t)[1]
            score = batch_mul(score, 1.0 / std)
        if residual_trick:
            # NOTE: so that if NN = 0 then time reversal = forward
            fwd_drift = sde.drift(y, t)
            residual = 2 * fwd_drift / sde.beta_schedule.beta_t(t)[..., None]
            score += residual
        if return_state:
            return score, new_state
        else:
            return score

    return score_fn
