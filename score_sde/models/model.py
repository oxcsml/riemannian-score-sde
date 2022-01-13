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
import math

import jax
import haiku as hk
import optax
import numpy as np
import jax.numpy as jnp

from score_sde.utils.jax import batch_mul
from score_sde.utils import register_category
from score_sde.utils.typing import ParametrisedScoreFunction

from score_sde.sde import SDE, VESDE, VPSDE, subVPSDE
from riemannian_score_sde.sde import Brownian
from .mlp import MLP


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

    if isinstance(sde, Brownian):

        def score_fn(x, t, rng=None):
            model_out, new_state = model.apply(params, state, rng, x=x, t=t)
            # NOTE: scaling the output with 1.0 / std helps cf 'Improved Techniques for Training Score-Based Generative Model'
            score = model_out
            std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
            score = batch_mul(model_out, 1.0 / std)
            if return_state:
                return score, new_state
            else:
                return score

    elif isinstance(sde, (VPSDE, subVPSDE)):

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


@dataclass
class Ignore:
    output_shape: int
    layer: object()

    def __call__(self, x, t):
        return self._layer(x)


@dataclass
class Concat:
    def __init__(self, output_shape, layer):
        self._layer = layer
        self._hyper_bias = MLP(
            hidden_shapes=[], output_shape=output_shape, act="", bias=False
        )

    def __call__(self, x, t):
        t = jnp.array(t, dtype=float).reshape(-1, 1)
        return self._layer(x) + self._hyper_bias(t)


@dataclass
class Squash:
    def __init__(self, output_shape, layer):
        self._layer = layer
        self._hyper = MLP(hidden_shapes=[], output_shape=output_shape, act="")

    def __call__(self, x, t):
        t = jnp.array(t, dtype=float).reshape(-1, 1)
        return self._layer(x) * jax.nn.sigmoid(self._hyper(t))


@dataclass
class ConcatSquash:
    def __init__(self, output_shape, layer):
        self._layer = layer
        self._hyper_bias = MLP(
            hidden_shapes=[], output_shape=output_shape, act="", bias=False
        )
        self._hyper_gate = MLP(hidden_shapes=[], output_shape=output_shape, act="")

    def __call__(self, x, t):
        t = jnp.array(t, dtype=float).reshape(-1, 1)
        return self._layer(x) * jax.nn.sigmoid(self._hyper_gate(t)) + self._hyper_bias(
            t
        )


def get_timestep_embedding(timesteps, embedding_dim=128):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=float) * -emb)

    emb = timesteps * jnp.expand_dims(emb, 0)
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [0, 1])

    return emb


@dataclass
class ScoreNetwork:
    def __init__(
        self, output_shape, encoder_layers=[16], pos_dim=16, decoder_layers=[128, 128]
    ):
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        # self.locals = [encoder_layers, pos_dim, decoder_layers, output_shape]

        self.net = MLP(
            hidden_shapes=decoder_layers, output_shape=output_shape, act="lrelu"
        )

        self.t_encoder = MLP(
            hidden_shapes=encoder_layers, output_shape=t_enc_dim, act="lrelu"
        )

        self.x_encoder = MLP(
            hidden_shapes=encoder_layers, output_shape=t_enc_dim, act="lrelu"
        )

    def __call__(self, x, t):
        t = jnp.array(t, dtype=float).reshape(-1, 1)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        temb = jnp.broadcast_to(temb, [xemb.shape[0], *temb.shape[1:]])
        h = jnp.concatenate([xemb, temb], -1)
        out = self.net(h)
        return out
