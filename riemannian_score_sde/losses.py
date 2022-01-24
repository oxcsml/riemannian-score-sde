"""All functions related to loss computation and optimization.
"""

from typing import Callable, Tuple
from functools import partial

import jax
import optax
import jax.numpy as jnp
import jax.random as random

from score_sde.sde import VESDE, VPSDE, SDE
from riemannian_score_sde.sde import Brownian
from score_sde.utils import batch_mul
from score_sde.models import get_score_fn, get_model_fn
from score_sde.utils import ParametrisedScoreFunction, TrainState
from score_sde.likelihood import div_noise, get_drift_fn, get_div_fn
from score_sde.sampling import get_pc_sampler


def get_ism_loss_fn(
    sde: SDE,
    model: ParametrisedScoreFunction,
    train: bool,
    reduce_mean: bool = True,
    likelihood_weighting: bool = True,
    hutchinson_type="Rademacher",
    eps: float = 1e-3,
):
    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        score_fn = get_score_fn(
            sde,
            model,
            params,
            states,
            train=train,
            continuous=continuous,
            return_state=True,
        )
        x_0 = batch["data"]

        rng, step_rng = random.split(rng)
        t = random.uniform(
            step_rng, (x_0.shape[0],), minval=sde.t0 + eps, maxval=sde.tf
        )

        rng, step_rng = random.split(rng)
        x_t = sde.marginal_sample(step_rng, x_0, t)
        score, new_model_state = score_fn(x_t, t, rng=step_rng)

        # ISM loss
        rng, step_rng = random.split(rng)
        epsilon = div_noise(step_rng, x_0.shape, hutchinson_type)
        drift_fn = lambda x, t: score_fn(x, t, rng=step_rng)[0]
        div_fn = get_div_fn(drift_fn, hutchinson_type)
        div_score = div_fn(x_t, t, epsilon)
        sq_norm_score = sde.manifold.metric.squared_norm(score, x_t)
        losses = 0.5 * sq_norm_score + div_score

        if likelihood_weighting:
            g2 = sde.coefficients(jnp.zeros_like(x_0), t)[1] ** 2
            losses = losses * g2

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn
