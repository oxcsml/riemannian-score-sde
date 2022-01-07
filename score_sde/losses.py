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
from functools import partial

import jax
import optax
import jax.numpy as jnp
import jax.random as random

from score_sde.sde import VESDE, VPSDE, SDE, Brownian
from score_sde.utils import batch_mul
from score_sde.models import get_score_fn, get_model_fn
from score_sde.utils import ParametrisedScoreFunction, TrainState
from score_sde.likelihood import p_div_fn, div_noise


def get_sde_loss_fn(
    sde: SDE,
    model: ParametrisedScoreFunction,
    train: bool,
    reduce_mean: bool = True,
    continuous: bool = True,
    likelihood_weighting: bool = True,
    eps: float = 1e-5,
    ism_loss: bool = False,
    hutchinson_type: str = "Rademacher",
) -> Callable[[jax.random.KeyArray, dict, dict, dict], Tuple[float, dict]]:
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde.SDE` object that represents the forward SDE.
      model: A transformed Haiku function object that represents the architecture of the score-based model.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        """Compute the loss function.

        Args:
          rng: A JAX random state.
          params: A dictionary that contains trainable parameters of the score-based model.
          states: A dictionary that contains mutable states of the score-based model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
          new_model_state: A dictionary that contains the mutated states of the score-based model.
        """

        score_fn = get_score_fn(
            sde,
            model,
            params,
            states,
            train=train,
            continuous=continuous,
            return_state=True,
        )
        data = batch["data"]

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
        rng, step_rng = random.split(rng)

        if isinstance(sde, Brownian):
            # TODO: problem if t is different for each batch value
            t = random.uniform(step_rng, (1,), minval=eps, maxval=sde.T)
            rng, step_rng = random.split(rng)
            perturbed_data = sde.marginal_sample(step_rng, data, t)
            t = jnp.ones(data.shape[0]) * t
            score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

            if not ism_loss:  # DSM loss
                logp_grad_fn = jax.value_and_grad(sde.marginal_log_prob, argnums=1, has_aux=False)
                logp, logp_grad = jax.vmap(logp_grad_fn)(data, perturbed_data, t)
            else:  # TODO: NOT tested!
                rng, step_rng = random.split(rng)
                epsilon = div_noise(step_rng, data.shape, hutchinson_type)
                logp_grad = p_div_fn(new_model_state, hutchinson_type, data, t, epsilon)

            losses = jnp.square(score - logp_grad)
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            if likelihood_weighting:
                raise NotImplementedError()
        else:
            t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
            rng, step_rng = random.split(rng)
            z = random.normal(step_rng, data.shape)
            mean, std = sde.marginal_prob(data, t)
            perturbed_data = mean + batch_mul(std, z)
            rng, step_rng = random.split(rng)
            score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

            if not likelihood_weighting:
                losses = jnp.square(batch_mul(score, std) + z)
                losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            else:
                g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
                losses = jnp.square(score + batch_mul(z, 1.0 / std))
                losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_smld_loss_fn(
    vesde: VESDE,
    model: ParametrisedScoreFunction,
    train: bool,
    reduce_mean: bool = False,
) -> Callable[[jax.random.KeyArray, dict, dict, dict], Tuple[float, dict]]:
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = vesde.discrete_sigmas[::-1]
    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        model_fn = get_model_fn(model, params, states, train=train)
        data = batch["image"]
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
        sigmas = smld_sigma_array[labels]
        rng, step_rng = random.split(rng)
        noise = batch_mul(random.normal(step_rng, data.shape), sigmas)
        perturbed_data = noise + data
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        target = -batch_mul(noise, 1.0 / (sigmas ** 2))
        losses = jnp.square(score - target)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * sigmas ** 2
        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_ddpm_loss_fn(
    vpsde: VPSDE,
    model: ParametrisedScoreFunction,
    train: bool,
    reduce_mean: bool = True,
) -> Callable[[jax.random.KeyArray, dict, dict, dict], Tuple[float, dict]]:
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        model_fn = get_model_fn(model, params, states, train=train)
        data = batch["image"]
        rng, step_rng = random.split(rng)
        labels = random.choice(step_rng, vpsde.N, shape=(data.shape[0],))
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, data.shape)
        perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + batch_mul(
            sqrt_1m_alphas_cumprod[labels], noise
        )
        rng, step_rng = random.split(rng)
        score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
        losses = jnp.square(score - noise)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_pmap_step_fn(
    sde: SDE,
    model: ParametrisedScoreFunction,
    optimizer,
    train: bool,
    reduce_mean=False,
    continuous=True,
    likelihood_weighting=False,
):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(
            sde,
            model,
            train,
            reduce_mean=reduce_mean,
            continuous=True,
            likelihood_weighting=likelihood_weighting,
        )
    else:
        assert (
            not likelihood_weighting
        ), "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(
                f"Discrete training for {sde.__class__.__name__} is not recommended."
            )

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
            (loss, new_model_state), grad = grad_fn(
                step_rng, params, model_state, batch
            )
            grad = jax.lax.pmean(grad, axis_name="batch")
            updates, new_opt_state = optimizer.update(grad, train_state.opt_state)
            new_parmas = optax.apply_updates(params, updates)

            new_params_ema = jax.tree_multimap(
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
                params_ema=new_params_ema,
                params=new_parmas,
            )
        else:
            loss, _ = loss_fn(
                step_rng, train_state.params_ema, train_state.model_state, batch
            )
            new_train_state = train_state

        loss = jax.lax.pmean(loss, axis_name="batch")
        new_carry_state = (rng, new_train_state)
        return new_carry_state, loss

    return step_fn


def get_step_fn(
    sde: SDE,
    model: ParametrisedScoreFunction,
    optimizer,
    train: bool,
    reduce_mean=False,
    continuous=True,
    likelihood_weighting=False,
):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A `flax.linen.Module` object that represents the architecture of the score-based model.
      train: `True` for training and `False` for evaluation.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(
            sde,
            model,
            train,
            reduce_mean=reduce_mean,
            continuous=True,
            likelihood_weighting=likelihood_weighting,
        )
    else:
        assert (
            not likelihood_weighting
        ), "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(
                f"Discrete training for {sde.__class__.__name__} is not recommended."
            )

    # @partial(jax.jit)
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
            (loss, new_model_state), grad = grad_fn(
                step_rng, params, model_state, batch
            )
            updates, new_opt_state = optimizer.update(grad, train_state.opt_state)
            new_parmas = optax.apply_updates(params, updates)

            new_params_ema = jax.tree_multimap(
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
