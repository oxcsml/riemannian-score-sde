import os
import logging
from functools import partial

from hydra.utils import instantiate, get_class, call
from omegaconf import OmegaConf

import jax
from jax import numpy as jnp
import numpy as np
import haiku as hk
import optax
from tqdm import tqdm

from score_sde.utils import TrainState, save, restore
from score_sde.utils.loggers_pl import LoggerCollection, Logger
from score_sde.utils.vis import plot, earth_plot
from score_sde.models import get_likelihood_fn_w_transform
from score_sde.datasets import random_split, DataLoader, TensorDataset
from score_sde.utils.tmp import compute_normalization
from score_sde.losses import get_ema_loss_step_fn

log = logging.getLogger(__name__)


def run(cfg):
    def train(train_state):
        loss = instantiate(
            cfg.loss, pushforward=pushforward, model=model, eps=cfg.eps, train=True
        )
        train_step_fn = get_ema_loss_step_fn(
            loss,
            optimizer=optimiser,
            train=True,
        )

        train_step_fn = jax.jit(train_step_fn)

        rng = train_state.rng
        t = tqdm(
            range(cfg.steps),
            total=cfg.steps,
            bar_format="{desc}{bar}{r_bar}",
            mininterval=1,
        )
        for step in t:
            batch = {"data": transform.inv(next(train_ds))}
            rng, next_rng = jax.random.split(rng)
            (rng, train_state), loss = train_step_fn((next_rng, train_state), batch)
            if jnp.isnan(loss).any():
                log.warning("Loss is nan")
                return train_state, False

            if step % 50 == 0:
                logger.log_metrics({"train/loss": loss}, step)
                t.set_description(f"Loss: {loss:.3f}")

            if step > 0 and step % cfg.val_freq == 0:
                save(ckpt_path, train_state)
                evaluate(train_state, "val", step)

        return train_state, True


    def evaluate(train_state, stage, step=None):
        log.info("Running evaluation")
        rng = jax.random.PRNGKey(cfg.seed)
        dataset = eval_ds if stage == "val" else test_ds

        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)
        # TODO: merge transorm as part of PushForward
        likelihood_fn = get_likelihood_fn_w_transform(likelihood_fn, transform)

        logp = 0.0
        N = 0

        if hasattr(dataset, "__len__"):
            for x in dataset:
                logp_step = likelihood_fn(rng, x)
                logp += logp_step.sum()
                N += logp_step.shape[0]
        else:
            # TODO: handle infinite datasets more elegnatly
            samples = 10
            for i in range(samples):
                x = next(dataset)
                logp_step = likelihood_fn(rng, x)
                logp += logp_step.sum()
                N += logp_step.shape[0]
        logp /= N

        logger.log_metrics({f"{stage}/logp": logp}, step)
        log.info(f"{stage}/logp = {logp:.3f}")
        if stage == "test":
            Z = compute_normalization(likelihood_fn)
            log.info(f"Z = {Z:.2f}")
            logger.log_metrics({f"{stage}/Z": Z}, step)


    def generate_plots(train_state, stage, step=None):
        log.info("Generating plots")
        rng = jax.random.PRNGKey(cfg.seed)
        dataset = eval_ds if stage == "eval" else test_ds

        x0 = next(dataset)
        z0 = transform.inv(x0)
        ## p_0 (backward)

        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        sampler = pushforward.get_sample(model_w_dicts, train=False)
        rng, next_rng = jax.random.split(rng)
        z = sampler(next_rng, z0.shape, N=100, eps=cfg.eps)
        x = transform(z)
        log.info(f"Prop samples in M: {100 * data_manifold.belongs(x, atol=1e-4).mean().item()}")

        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)
        # TODO: merge transorm as part of PushForward
        likelihood_fn = get_likelihood_fn_w_transform(likelihood_fn, transform)
        likelihood_fn = partial(likelihood_fn, rng)

        plt = plot(None, x, jnp.exp(likelihood_fn(x)), None)
        logger.log_plot("x0_backw", plt, cfg.steps)
        # prob = jnp.exp(dataset.log_prob(x0)) if hasattr(dataset, "log_prob") else None
        # plt = plot(None, x0, prob, None)
        # logger.log_plot("x0_true", plt, cfg.steps)

        plt = earth_plot(cfg, likelihood_fn, train_ds, test_ds, N=500, samples=x)
        if plt is not None:
            logger.log_plot("pdf", plt, cfg.steps)


    ### Main
    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    ckpt_path = os.path.join(run_path, cfg.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    Logger.instance().set_logger(logger)
    # Logger.get() -> returns global logger anywhere from here

    # TODO: should sample random seed given a run id?
    rng = jax.random.PRNGKey(cfg.seed)
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    flow = instantiate(cfg.flow, manifold=model_manifold)
    pushforward = instantiate(cfg.pushf, model_manifold, flow)

    log.info("Stage : Instantiate dataset")

    rng, next_rng = jax.random.split(rng)
    # dataset = instantiate(cfg.dataset, rng=next_rng, manifold=data_manifold)
    # TODO: Handle infinate datasets more elegantly?
    try:
        dataset = instantiate(cfg.dataset, rng=next_rng)
    except:
        dataset = instantiate(cfg.dataset)

    if isinstance(dataset, TensorDataset):
        train_ds, eval_ds, test_ds = random_split(
            dataset, lengths=cfg.splits, rng=next_rng
        )
        train_ds, eval_ds, test_ds = (
            DataLoader(
                train_ds,
                batch_dims=cfg.batch_size,
                rng=next_rng,
                shuffle=True,
                drop_last=False,
            ),
            DataLoader(
                eval_ds,
                batch_dims=cfg.batch_size,
                rng=next_rng,
                shuffle=True,
                drop_last=False,
            ),
            DataLoader(
                test_ds,
                batch_dims=cfg.batch_size,
                rng=next_rng,
                shuffle=True,
                drop_last=False,
            ),
        )
        log.info(
            f"Train size: {len(train_ds.dataset)}. Val size: {len(eval_ds.dataset)}. Test size: {len(test_ds.dataset)}"
        )
    else:
        train_ds, eval_ds, test_ds = dataset, dataset, dataset

    z = transform.inv(next(train_ds))

    log.info("Stage : Instantiate model")

    def model(x, t):
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=model_manifold,
        )
        return score(x, t)

    model = hk.transform_with_state(model)

    rng, next_rng = jax.random.split(rng)
    params, state = model.init(rng=next_rng, x=z, t=0)

    log.info("Stage : Instantiate optimiser")

    schedule_fn = instantiate(cfg.scheduler)
    optimiser = optax.chain(
        instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn)
    )
    opt_state = optimiser.init(params)

    if cfg.resume or cfg.mode == "test":  # if resume or evaluate
        train_state = restore(ckpt_path)
    else:
        rng, next_rng = jax.random.split(rng)
        train_state = TrainState(
            opt_state=opt_state,
            model_state=state,
            step=0,
            params=params,
            ema_rate=cfg.ema_rate,
            params_ema=params,
            rng=next_rng,  # TODO: we should actually use this for reproducibility
        )

    if cfg.mode == "train" or cfg.mode == "all":
        log.info("Stage : Training")
        train_state, success = train(train_state)
    if (cfg.mode == "test") or (cfg.mode == "all" and success):
        log.info("Stage : Test")
        evaluate(train_state, "test")
        generate_plots(train_state, "test")
        success = True
    logger.save()
    logger.finalize("success" if success else "failure")
