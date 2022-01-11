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
"""Common layers for defining score networks.
"""
import functools
import math
import string
from typing import Any, Sequence, Optional
from dataclasses import dataclass

import jax
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp

from score_sde.utils import register_category

get_activation, register_activation = register_category("activation")

register_activation(jnn.elu, name="elu")
register_activation(jnn.relu, name="relu")
register_activation(functools.partial(jnn.leaky_relu, negative_slope=0.01), name="lrelu")
register_activation(jnn.swish, name="swish")
register_activation(jnp.sin, name='sin')


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return hk.initializers.VarianceScaling(scale, "fan_avg", "uniform")


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

@dataclass
class NIN(hk.Module):
    num_units: int
    init_scale: float = 0.1

    def __call__(self, x):
        in_dim = int(x.shape[-1])
        W = hk.get_parameter(
            "w", shape=(in_dim, self.num_units), dtype=x.dtype, init=default_init()
        )
        b = hk.get_parameter(
            "b",
            shape=(self.num_units,),
            dtype=x.dtype,
            init=hk.initializers.Constant(0.0),
        )
        y = contract_inner(x, W) + b
        assert y.shape == x.shape[:-1] + (self.num_units,)
        return y


def _einsum(a, b, c, x, y):
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return jnp.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_uppercase[: len(y.shape)])
    assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


@dataclass
class AttentionBlock(hk.Module):
    """Channel-wise self-attention block."""

    normalize: Any

    def __call__(self, x):
        B, H, W, C = x.shape
        h = self.normalize()(x)
        q = NIN(C)(h)
        k = NIN(C)(h)
        v = NIN(C)(h)

        w = jnp.einsum("bhwc,bHWc->bhwHW", q, k) * (int(C) ** (-0.5))
        w = jnp.reshape(w, (B, H, W, H * W))
        w = jax.nn.softmax(w, axis=-1)
        w = jnp.reshape(w, (B, H, W, H, W))
        h = jnp.einsum("bhwHW,bHWc->bhwc", w, v)
        h = NIN(C, init_scale=0.0)(h)
        return x + h
