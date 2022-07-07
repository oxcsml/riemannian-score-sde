from dataclasses import dataclass
import math

import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp

from .mlp import MLP


@dataclass
class Concat(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)

    def __call__(self, x, t):
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
        self._hyper_bias = MLP(
            hidden_shapes=[], output_shape=output_shape, act="", bias=False
        )

    def __call__(self, x, t):
        t = jnp.array(t, dtype=float).reshape(-1, 1)
        return self._layer(x) + self._hyper_bias(t)


@dataclass
class Squash(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)
        self._hyper = MLP(hidden_shapes=[], output_shape=output_shape, act="")

    def __call__(self, x, t):
        t = jnp.array(t, dtype=float).reshape(-1, 1)
        return self._layer(x) * jax.nn.sigmoid(self._hyper(t))


@dataclass
class SquashSum(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)
        self._hyper_bias = MLP(
            hidden_shapes=[], output_shape=output_shape, act="", bias=False
        )
        self._hyper_gate = MLP(hidden_shapes=[], output_shape=output_shape, act="")

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
        emb = jnp.pad(emb, [0, 1])

    return emb


@dataclass
class ConcatEmbed(hk.Module):
    def __init__(
        self,
        output_shape,
        enc_shapes,
        t_dim,
        dec_shapes,
        act,
    ):
        super().__init__()
        self.temb_dim = t_dim
        t_enc_dim = t_dim * 2

        self.net = MLP(hidden_shapes=dec_shapes, output_shape=output_shape, act=act)

        self.t_encoder = MLP(hidden_shapes=enc_shapes, output_shape=t_enc_dim, act=act)

        self.x_encoder = MLP(hidden_shapes=enc_shapes, output_shape=t_enc_dim, act=act)

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
