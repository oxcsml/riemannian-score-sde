import jax
import optax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves

from score_sde.utils import TrainState


def build_optimize_fn(warmup: bool, grad_clip: float):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(state: TrainState, grad: dict, warmup=warmup, grad_clip=grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        lr = state.lr
        if warmup > 0:
            lr = lr * jnp.minimum(state.step / warmup, 1.0)
        if grad_clip >= 0:
            # Compute global gradient norm
            grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in tree_leaves(grad)]))
            # Clip gradient
            clipped_grad = tree_map(
                lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad
            )
        else:  # disabling gradient clipping if grad_clip < 0
            clipped_grad = grad
        return state.optimizer.apply_updates(clipped_grad, learning_rate=lr)

    return optimize_fn
