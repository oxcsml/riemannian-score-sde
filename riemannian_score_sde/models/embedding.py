import abc
from dataclasses import dataclass

import jax
import haiku as hk
import jax.numpy as jnp

from score_sde.models import MLP


class Embedding(hk.Module, abc.ABC):
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold


class NoneEmbedding(Embedding):
    def __call__(self, x, t):
        return x, t


class LaplacianEigenfunctionEmbedding(Embedding):
    def __init__(self, manifold, n_manifold, n_time, max_t):
        super().__init__(manifold)
        self.n_time = n_time
        self.frequencies = 2 * jnp.pi * jnp.arange(n_time) / max_t
        self.n_manifold = n_manifold

    def __call__(self, x, t):
        t = jnp.array(t)
        if len(t.shape) == 0:
            t = t * jnp.ones(x.shape[:-1])

        if len(t.shape) == len(x.shape) - 1:
            t = jnp.expand_dims(t, axis=-1)

        x = self.manifold.laplacian_eigenfunctions(x, self.n_time)
        t = jnp.concatenate(
            (jnp.cos(self.frequencies * t), jnp.sin(self.frequencies * t)), axis=-1
        )

        return x, t


@dataclass
class ConcatEigenfunctionEmbed(hk.Module):
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
