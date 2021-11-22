from typing import Callable

import jax
import numpy as np
import jax.numpy as jnp

from .typing import ScoreFunction


def batch_add(a, b):
    return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def get_div_fn(fn: ScoreFunction):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x: jnp.ndarray, t: float, eps: jp.ndarray):
        grad_fn = lambda data: jnp.sum(fn(data, t) * eps)
        grad_fn_eps = jax.grad(grad_fn)(x)
        return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

    return div_fn


def to_flattened_numpy(x: jnp.ndarray) -> np.ndarray:
    """Flatten a JAX array `x` and convert it to numpy."""
    return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x: np.ndarray, shape: tuple) -> jnp.ndarray:
    """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
    return jnp.asarray(x).reshape(shape)
