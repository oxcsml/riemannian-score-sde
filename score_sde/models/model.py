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
from typing import Any, NamedTuple, Callable, Sequence
import math
from geomstats.geometry.hypersphere import Hypersphere
from hydra.utils import instantiate
import abc

import jax
import haiku as hk
import optax
import numpy as np
import jax.numpy as jnp

from score_sde.utils.jax import batch_mul
from score_sde.utils import register_category
from score_sde.utils.typing import ParametrisedScoreFunction

from score_sde.sde import SDE, VESDE, VPSDE, subVPSDE, Brownian
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
class Concat(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)

    def __call__(self, x, t):
        t = jnp.array(t)
        if len(t.shape) == 0:
            t = t * jnp.ones(x.shape[:-1])

        if len(t.shape) == len(x.shape) - 1:
            t = jnp.expand_dims(t, axis=-1)

        return self._layer(jnp.concatenate([x, t], axis=-1))


@dataclass
class Ignore(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)

    def __call__(self, x, t):
        return self._layer(x)


@dataclass
class Sum(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)
        self._hyper_bias = MLP(hidden_shapes=[], output_shape=output_shape, act='', bias=False)

    def __call__(self, x, t):
        t = jnp.array(t, dtype=float).reshape(-1, 1)
        return self._layer(x) + self._hyper_bias(t)


@dataclass
class Squash(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)
        self._hyper = MLP(hidden_shapes=[], output_shape=output_shape, act='')

    def __call__(self, x, t):
        t = jnp.array(t, dtype=float).reshape(-1, 1)
        return self._layer(x) * jax.nn.sigmoid(self._hyper(t))


@dataclass
class SquashSum(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)
        self._hyper_bias = MLP(hidden_shapes=[], output_shape=output_shape, act='', bias=False)
        self._hyper_gate = MLP(hidden_shapes=[], output_shape=output_shape, act='')

    def __call__(self, x, t):
        t = jnp.array(t, dtype=float).reshape(-1, 1)
        return self._layer(x) * jax.nn.sigmoid(self._hyper_gate(t)) + self._hyper_bias(t)


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
        emb = jnp.pad(emb, [0,1])

    return emb


@dataclass
class ConcatEmbed(hk.Module):
    def __init__(self, output_shape, encoder_layers=[16], pos_dim=16, decoder_layers=[128,128], act='lrelu'):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2

        self.net = MLP(hidden_shapes=decoder_layers,
                       output_shape=output_shape,
                       act=act)

        self.t_encoder = MLP(hidden_shapes=encoder_layers,
                             output_shape=t_enc_dim,
                             act=act)

        self.x_encoder = MLP(hidden_shapes=encoder_layers,
                             output_shape=t_enc_dim,
                             act=act)

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


def get_div_fn(fi_fn, Xi, hutchinson_type: str):
    """Pmapped divergence of the drift function."""
    if hutchinson_type == "None":
        return lambda x, t, eps: get_exact_div_fn(fi_fn, Xi)(x, t)
    else:
        return lambda x, t, eps: get_estimate_div_fn(fi_fn, Xi)(x, t, eps)


def get_estimate_div_fn(fi_fn, Xi=None):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x: jnp.ndarray, t: float, eps: jnp.ndarray):
        # grad_fn = lambda data: jnp.sum(fi_fn(data, t) * eps)
        # grad_fn_eps = jax.grad(grad_fn)(x)
        def grad_fn(data):
            fi = fi_fn(data, t)
            return jnp.sum(fi * eps), fi
        (_, fi), grad_fn_eps = jax.value_and_grad(grad_fn, has_aux=True)(x)
        # out = grad_fn_eps * G(x) @ Xi * eps
        # G = manifold.metric.metric_matrix(x)
        # Xi = jnp.einsum('...ij,...jk->...ik', G, Xi)
        if Xi is not None:
            grad_fn_eps = jnp.einsum('...d,...dn->...n', grad_fn_eps, Xi)
        div = jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))
        return div, fi

    return div_fn


def get_exact_div_fn(fi_fn, Xi=None):
    "flatten all but the last axis and compute the true divergence"

    def div_fn(
        x: jnp.ndarray,
        t: float,
    ):
        if len(t.shape) == len(x.shape) - 1:
            # Assume t is just missing the last dim of x
            t = jnp.expand_dims(t, axis=-1)

        x_shape = x.shape
        x = jnp.expand_dims(x.reshape((-1, x_shape[-1])), 1)
        t = jnp.expand_dims(t.reshape((-1, t.shape[-1])), 1)
        jac = jax.vmap(jax.jacrev(fi_fn, argnums=0))(x, t)
        jac = jac.reshape([*x_shape[:-1], -1, x_shape[-1]])
        # G = manifold.metric.metric_matrix(x)
        # Xi = jnp.einsum('...ij,...jk->...ik', G, Xi(x))
        if Xi is not None:
            jac = jnp.einsum('...nd,...dm->...nm', jac, Xi)
        div = jnp.trace(jac, axis1=-1, axis2=-2).reshape(x_shape[:-1])
        return div

    return div_fn


class VectorFieldGenerator(hk.Module, abc.ABC):
    def __init__(self, architecture, output_shape, manifold):
        """X = fi * Xi with fi weights and Xi generators"""
        super().__init__()
        self.net = instantiate(architecture, output_shape=output_shape)
        self.manifold = manifold

    @staticmethod
    @abc.abstractmethod
    def output_shape(manifold):
        """Cardinality of the generating set."""

    def _weights(self, x, t):
        """shape=[..., card=n]"""
        return self.net(x, t)

    @abc.abstractmethod
    def _generators(self, x):
        """Set of generating vector fields: shape=[..., d, card=n]"""

    @property
    def decomposition(self):
        return lambda x, t: self._weights(x, t), lambda x: self._generators(x)

    def __call__(self, x, t):
        fi_fn, Xi_fn = self.decomposition
        fi, Xi = fi_fn(x, t), Xi_fn(x)
        out = jnp.einsum('...n,...dn->...d', fi, Xi)
        # assert self.manifold.is_tangent(out, x, atol=1e-6).all()
        return out

    def div_generators(self, x):
        """Divergence of the generating vector fields: shape=[..., card=n]"""

    def div_split(self, x, t, hutchinson_type):
        """Returns div(X) = Xi(fi) + fi div(Xi)"""
        fi_fn, Xi_fn = self.decomposition
        Xi = Xi_fn(x)
        if hutchinson_type == "None":
            # splitting div is unecessary when computated exactly
            # term_1 = get_exact_div_fn(fi_fn, Xi)(x, t)
            out = get_exact_div_fn(self.__call__, None)(x, t)
        else:
            shape = [*x.shape[:-1], self.output_shape(self.manifold)]
            eps = div_noise(hk.next_rng_key(), shape, hutchinson_type)
            term_1, fi = get_estimate_div_fn(fi_fn, Xi)(x, t, eps)
            div_Xi = self.div_generators(x)
            term_2 = jnp.einsum('...n,...n->...', fi, div_Xi)
            out = term_1 + term_2
        return out

    def divE(self, x, t, hutchinson_type):
        """Euclidean divergence cf Rozen et al. 2021"""
        if hutchinson_type == "None":
            out = get_exact_div_fn(self.__call__, None)(x, t)
        else:
            shape = [*x.shape[:-1], self.output_shape(self.manifold)]
            eps = div_noise(hk.next_rng_key(), shape, hutchinson_type)
            out, _ = get_estimate_div_fn(self.__call__, None)(x, t, eps)
        return out


class DivFreeGenerator(VectorFieldGenerator):
    def __init__(self, architecture, output_shape, manifold):
        super().__init__(architecture, output_shape, manifold)
        self.div = self.div_split
        # self.div = self.divE
    
    @staticmethod
    def output_shape(manifold):
        return manifold.isom_group.dim

    def _generators(self, x):
        return self.manifold.div_free_generators(x)

    def div_generators(self, x):
        shape = [*x.shape[:-1], self.output_shape(self.manifold)]
        return jnp.zeros(shape)


class EigenGenerator(VectorFieldGenerator):
    """Gradient of laplacien eigenfunctions with eigenvalue=1"""
    def __init__(self, architecture, output_shape, manifold):
        super().__init__(architecture, output_shape, manifold)
        assert isinstance(manifold, Hypersphere)
        self.div = self.div_split

    @staticmethod
    def output_shape(manifold):
        return manifold.embedding_space.dim

    def _generators(self, x):
        return self.manifold.eigen_generators(x)

    def div_generators(self, x):
        # NOTE: Empirically need this factor 2 but why??
        return - self.manifold.dim * 2 * x


class AmbientGenerator(VectorFieldGenerator):
    """Equivalent to EigenGenerator"""
    def __init__(self, architecture, output_shape, manifold):
        super().__init__(architecture, output_shape, manifold)
        self.div = self.divE
    
    @staticmethod
    def output_shape(manifold):
        return manifold.embedding_space.dim
    
    def _generators(self, x):
        return self.manifold.eigen_generators(x)

    # def __call__(self, x, t):
    #     # `to_tangent`` have an 1/sq_norm(x) term that wrongs the div
    #     return self.manifold.to_tangent(self.net(x, t), x)
