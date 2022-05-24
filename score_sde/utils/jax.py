import os
import pickle
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

    def div_fn(x: jnp.ndarray, t: float, z: jnp.ndarray, eps: jnp.ndarray):
        eps = eps.reshape(eps.shape[0], -1)
        grad_fn = lambda x: jnp.sum(fn(x, t, z) * eps)
        grad_fn_eps = jax.grad(grad_fn)(x).reshape(x.shape[0], -1)
        # grad_fn = lambda x, t, eps: jnp.sum(fn(x, t) * eps)
        # grad_fn_eps = jax.vmap(jax.grad(grad_fn, argnums=0))(x, t, eps)
        return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(eps.shape))))

    return div_fn


def get_exact_div_fn(fn):
    "flatten all but the last axis and compute the true divergence"

    def div_fn(x: jnp.ndarray, t: float, z: jnp.ndarray):
        x_shape = x.shape
        dim = np.prod(x_shape[1:])
        # x = x.reshape((x.shape[0], -1))  # vec(x)
        t = jnp.expand_dims(t.reshape(-1), axis=-1)
        # x = jnp.expand_dims(x.reshape((-1, x_shape[-1])), 1)
        # t = jnp.expand_dims(t.reshape((-1, t_shape[-1])), 1)
        x = jnp.expand_dims(x, 1)  # NOTE: need leading batch dim after vmap
        if z is not None:
            z = jnp.expand_dims(z, 1)
        t = jnp.expand_dims(t, 1)
        jac = jax.vmap(jax.jacrev(fn, argnums=0))(x, t, z)
        # jac = jac.reshape([*x_shape[:-1], x_shape[-1], x_shape[-1]])

        jac = jac.reshape([x_shape[0], dim, dim])
        return jnp.trace(jac, axis1=-1, axis2=-2)  # .reshape(x_shape[:-1])

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


def save(ckpt_dir: str, state) -> None:
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def restore(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_unflatten(treedef, flat_state)
