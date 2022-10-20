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

"""All functions related to loss computation and optimization.
"""

from typing import Callable, Tuple

import jax
import optax
import jax.numpy as jnp
import jax.random as random
from jax.tree_util import tree_map

from score_sde.utils import batch_mul
from score_sde.models import PushForward, SDEPushForward
from score_sde.utils import ParametrisedScoreFunction, TrainState
from score_sde.models import div_noise, get_div_fn


def get_dsm_loss_fn(
    pushforward: SDEPushForward,
    model: ParametrisedScoreFunction,
    train: bool = True,
    reduce_mean: bool = True,
    like_w: bool = True,
    eps: float = 1e-3,
):
    sde = pushforward.sde
    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        score_fn = sde.reparametrise_score_fn(model, params, states, train, True)
        x_0 = batch["data"]

        rng, step_rng = random.split(rng)
        # uniformly sample from SDE timeframe
        t = random.uniform(step_rng, (x_0.shape[0],), minval=sde.t0 + eps, maxval=sde.tf)
        rng, step_rng = random.split(rng)
        z = random.normal(step_rng, x_0.shape)
        mean, std = sde.marginal_prob(x_0, t)
        # reparametrised sample x_t|x_0 = mean + std * z with z ~ N(0,1)
        x_t = mean + batch_mul(std, z)
        score, new_model_state = score_fn(x_t, t, rng=step_rng)
        # grad log p(x_t|x_0) = - 1/std^2 (x_t - mean) = - z / std

        if not like_w:
            losses = jnp.square(batch_mul(score, std) + z)
            # losses = std^2 * DSM(x_t, x_0)
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        else:
            g2 = sde.coefficients(jnp.zeros_like(x_0), t)[1] ** 2
            # losses = DSM(x_t, x_0)
            losses = jnp.square(score + batch_mul(z, 1.0 / std))
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_ism_loss_fn(
    pushforward: SDEPushForward,
    model: ParametrisedScoreFunction,
    train: bool,
    like_w: bool = True,
    hutchinson_type="Rademacher",
    eps: float = 1e-3,
):
    sde = pushforward.sde

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        score_fn = sde.reparametrise_score_fn(model, params, states, train, True)
        x_0 = batch["data"]

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (x_0.shape[0],), minval=sde.t0 + eps, maxval=sde.tf)

        rng, step_rng = random.split(rng)
        x_t = sde.marginal_sample(step_rng, x_0, t)
        score, new_model_state = score_fn(x_t, t, rng=step_rng)

        # ISM loss
        rng, step_rng = random.split(rng)
        epsilon = div_noise(step_rng, x_0.shape, hutchinson_type)
        drift_fn = lambda x, t: score_fn(x, t, rng=step_rng)[0]
        div_fn = get_div_fn(drift_fn, hutchinson_type)
        div_score = div_fn(x_t, t, epsilon)
        sq_norm_score = jnp.power(score, 2).sum(axis=-1)
        losses = 0.5 * sq_norm_score + div_score

        if like_w:
            g2 = sde.coefficients(jnp.zeros_like(x_0), t)[1] ** 2
            losses = losses * g2

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_logp_loss_fn(
    pushforward: PushForward,
    model: ParametrisedScoreFunction,
    train: bool = True,
    **kwargs
):
    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        x_0 = batch["data"]
        context = batch["context"]

        model_w_dicts = (model, params, states)
        log_prob = pushforward.get_log_prob(model_w_dicts, train=train)
        losses = -log_prob(x_0, context, rng=rng)[0]
        loss = jnp.mean(losses)

        return loss, states

    return loss_fn


def get_ema_loss_step_fn(
    loss_fn,
    optimizer,
    train: bool,
):
    """Create a one-step training/evaluation function.

    Args:
      loss_fn: loss function to compute
      train: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      like_w: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """

    def step_fn(carry_state: Tuple[jax.random.KeyArray, TrainState], batch: dict):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, NamedTuple containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, train_state) = carry_state
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        if train:
            params = train_state.params
            model_state = train_state.model_state
            (loss, new_model_state), grad = grad_fn(step_rng, params, model_state, batch)
            updates, new_opt_state = optimizer.update(grad, train_state.opt_state)
            new_parmas = optax.apply_updates(params, updates)

            new_params_ema = tree_map(
                lambda p_ema, p: p_ema * train_state.ema_rate
                + p * (1.0 - train_state.ema_rate),
                train_state.params_ema,
                new_parmas,
            )
            step = train_state.step + 1
            new_train_state = train_state._replace(
                step=step,
                opt_state=new_opt_state,
                model_state=new_model_state,
                params=new_parmas,
                params_ema=new_params_ema,
            )
        else:
            loss, _ = loss_fn(
                step_rng, train_state.params_ema, train_state.model_state, batch
            )
            new_train_state = train_state

        new_carry_state = (rng, new_train_state)
        return new_carry_state, loss

    return step_fn
