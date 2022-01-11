from typing import Callable

import jax
import numpy as np
import jax.numpy as jnp
import jax.lib.xla_bridge as xb


from .typing import ScoreFunction


def batch_add(a, b):
    return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def get_estimate_div_fn(fn: ScoreFunction):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x: jnp.ndarray, t: float, eps: jnp.ndarray):
        grad_fn = lambda data: jnp.sum(fn(data, t) * eps)
        grad_fn_eps = jax.grad(grad_fn)(x)
        return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))

    return div_fn


def get_exact_div_fn(fn):
    "flatten all but the last axis and compute the true divergence"

    def div_fn(
        x: jnp.ndarray,
        t: float,
    ):
        if len(t.shape) == len(x.shape) - 1:
            # Assume t is just missing the last dim of x
            t = jnp.expand_dims(t, axis=-1)

        x_shape = x.shape
        t_shape = t.shape

        x = x.reshape((-1, x_shape[-1]))
        t = t.reshape((-1, t_shape[-1]))
        jac = jax.vmap(jax.jacrev(fn, argnums=0))(x, t)
        return jnp.trace(jac, axis1=-1, axis2=-2).reshape(x_shape[:-1])

    return div_fn


def to_flattened_numpy(x: jnp.ndarray) -> np.ndarray:
    """Flatten a JAX array `x` and convert it to numpy."""
    return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x: np.ndarray, shape: tuple) -> jnp.ndarray:
    """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
    return jnp.asarray(x).reshape(shape)


# Borrowed from flax
def replicate(tree, devices=None):
    """Replicates arrays to multiple devices.

    Args:
      tree: a pytree containing the arrays that should be replicated.
      devices: the devices the data is replicated to
        (default: `jax.local_devices()`).
    Returns:
      A new pytree containing the replicated arrays.
    """
    if devices is None:
        # match the default device assignments used in pmap:
        # for single-host, that's the XLA default device assignment
        # for multi-host, it's the order of jax.local_devices()
        if jax.process_count() == 1:
            devices = [
                d
                for d in xb.get_backend().get_default_device_assignment(
                    jax.device_count()
                )
                if d.process_index == jax.process_index()
            ]
        else:
            devices = jax.local_devices()

    return jax.device_put_replicated(tree, devices)


# Borrowed from flax
def unreplicate(tree):
    """Returns a single instance of a replicated array."""
    return jax.tree_map(lambda x: x[0], tree)
