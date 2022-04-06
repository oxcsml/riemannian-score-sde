import os
import logging
from functools import partial
from timeit import default_timer as timer
import socket

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
from score_sde.datasets import (
    random_split,
    DataLoader,
    TensorDataset,
    get_data_per_context,
)
from score_sde.utils.normalization import compute_normalization
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
            range(train_state.step, cfg.steps),
            total=cfg.steps - train_state.step,
            bar_format="{desc}{bar}{r_bar}",
            mininterval=1,
        )
        train_time = timer()
        total_train_time = 0
        for step in t:
            data, z = next(train_ds)
            batch = {"data": transform.inv(data), "context": z}
            rng, next_rng = jax.random.split(rng)
            (rng, train_state), loss = train_step_fn((next_rng, train_state), batch)
            if jnp.isnan(loss).any():
                log.warning("Loss is nan")
                return train_state, False

            if step % 50 == 0:
                logger.log_metrics({"train/loss": loss}, step)
                t.set_description(f"Loss: {loss:.3f}")

            if step > 0 and step % cfg.val_freq == 0:
                logger.log_metrics(
                    {"train/time_per_it": (timer() - train_time) / cfg.val_freq}, step
                )
                total_train_time += timer() - train_time
                save(ckpt_path, train_state)
                eval_time = timer()
                evaluate(train_state, "val", step)
                logger.log_metrics({"val/time_per_it": (timer() - eval_time)}, step)
                train_time = timer()

        logger.log_metrics({"train/total_time": total_train_time}, step)
        return train_state, True

    def evaluate(train_state, stage, step=None):
        log.info("Running evaluation")
        rng = jax.random.PRNGKey(cfg.seed)
        dataset = eval_ds if stage == "val" else test_ds

        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        likelihood_fn_wo_tr = pushforward.get_log_prob(model_w_dicts, train=False)
        likelihood_fn_wo_tr = partial(likelihood_fn_wo_tr, rng)
        # TODO: merge transorm as part of PushForward
        likelihood_fn = get_likelihood_fn_w_transform(likelihood_fn_wo_tr, transform)

        logp = 0.0
        N = 0

        if hasattr(dataset, "__len__"):
            for batch in dataset:
                logp_step = likelihood_fn(*batch)
                logp += logp_step.sum()
                N += logp_step.shape[0]
        else:
            # TODO: handle infinite datasets more elegnatly
            samples = 10
            for i in range(samples):
                batch = next(dataset)
                logp_step = likelihood_fn(*batch)
                logp += logp_step.sum()
                N += logp_step.shape[0]
        logp /= N

        logger.log_metrics({f"{stage}/logp": logp}, step)
        log.info(f"{stage}/logp = {logp:.3f}")

        if stage == "test":
            default_z = z[0] if z is not None else None
            Z = compute_normalization(likelihood_fn, data_manifold, z=default_z)
            log.info(f"Z = {Z:.2f}")
            logger.log_metrics({f"{stage}/Z": Z}, step)

    def generate_plots(train_state, stage, step=None):
        log.info("Generating plots")
        rng = jax.random.PRNGKey(cfg.seed)
        dataset = eval_ds if stage == "eval" else test_ds

        M = 32
        # M = 2
        x0, y0 = get_data_per_context(dataset, transform, M)
        ## p_0 (backward)

        rng, next_rng = jax.random.split(rng)
        model_w_dicts = (model, train_state.params_ema, train_state.model_state)

        sampler = pushforward.get_sample(model_w_dicts, train=False)
        likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)

        z = next(dataset)[1]
        unique_z = [None] if z is None else jnp.unique(z).reshape((-1, 1))
        xs = []
        shape = (int(cfg.batch_size * M / len(unique_z)), *y0.shape[1:])
        for k, z in enumerate(unique_z):
            y = sampler(next_rng, shape, z, N=100, eps=cfg.eps)
            xs.append(transform(y))
            log.info(
                f"Prop samples in M: {100 * data_manifold.belongs(xs[-1], atol=1e-4).mean().item()}"
            )

            likelihood_fn = get_likelihood_fn_w_transform(likelihood_fn, transform)
            likelihood_fn = partial(likelihood_fn, rng, z=z)
            # plt = earth_plot(cfg, likelihood_fn, train_ds, test_ds, N=500, samples=x)
            # if plt is not None:
            #     logger.log_plot(f"pdf_{k}", plt, cfg.steps)

        plt = plot(data_manifold, x0, xs)  # prob=jnp.exp(likelihood_fn(x)
        logger.log_plot(f"x0_backw", plt, cfg.steps)

    ### Main
    # jax.config.update("jax_enable_x64", True)
    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
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

    log.info("Stage : Instantiate model")

    def model(x, t, z):
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=model_manifold,
        )
        if z is not None:
            t_expanded = jnp.expand_dims(t.reshape(-1), -1)
            if z.shape[0] != x.shape[0]:
                z = jnp.repeat(jnp.expand_dims(z, 0), x.shape[0], 0)
            z = jnp.concatenate([t_expanded, z], axis=-1)
        else:
            z = t
        return score(x, z)

    model = hk.transform_with_state(model)

    rng, next_rng = jax.random.split(rng)
    t = jnp.zeros((cfg.batch_size, 1))
    data, z = next(train_ds)
    y = transform.inv(data)
    params, state = model.init(rng=next_rng, x=y, t=t, z=z)

    log.info("Stage : Instantiate optimiser")

    schedule_fn = instantiate(cfg.scheduler)
    optimiser = optax.chain(instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn))
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
