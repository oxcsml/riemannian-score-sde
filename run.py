from functools import partial
from pathlib import Path
import logging
import hydra
from hydra.utils import instantiate, get_class

import jax
from jax import numpy as jnp
import numpy as np
import haiku as hk
import optax

from score_sde.utils import TrainState, save, restore
from score_sde.sampling import EulerMaruyamaManifoldPredictor, get_pc_sampler
from score_sde.likelihood import get_likelihood_fn
from score_sde.utils.vis import plot_and_save

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main")
def run(cfg):
    log.info("Stage : Startup")
    # print(cfg)

    rng = jax.random.PRNGKey(cfg.seed)
    manifold = instantiate(cfg.manifold)
    sde = instantiate(cfg.sde, manifold)

    log.info("Stage : Instantiate dataset")

    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng, manifold=manifold)
    x = next(dataset)

    log.info("Stage : Instantiate model")

    output_shape = get_class(cfg.generator._target_).output_shape(manifold)

    # def score_model(x, t):
    #     score = instantiate(cfg.generator, cfg.architecture, output_shape, manifold=manifold)
    #     return score(x, t)

    def score_model(x, t, div=False, hutchinson_type='None'):
        score = instantiate(cfg.generator, cfg.architecture, output_shape, manifold=manifold)
        if not div:
            return score(x, t)
        else:
            return score.div(x, t, hutchinson_type)

    score_model = hk.transform_with_state(score_model)

    rng, next_rng = jax.random.split(rng)
    params, state = score_model.init(rng=next_rng, x=x, t=0)
    # out, _ = score_model.apply(params, state, next_rng, x=x, t=0)
    # print(out.shape)
    hutchinson_type='Rademacher'
    # hutchinson_type='None'
    rng, step_rng = jax.random.split(rng)
    t = jax.random.uniform(step_rng, (x.shape[0],), minval=cfg.eps, maxval=sde.T)
    # out, _ = score_model.apply(params, state, next_rng, x=x, t=t, div=True, hutchinson_type=hutchinson_type)
    # print(out.shape)
    # raise

    log.info("Stage : Instantiate optimiser")

    schedule_fn = instantiate(cfg.scheduler)
    optimiser = optax.chain(instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn))
    opt_state = optimiser.init(params)

    rng, next_rng = jax.random.split(rng)
    train_state = TrainState(
        opt_state=opt_state, model_state=state, step=0, params=params, ema_rate=cfg.ema_rate, params_ema=params, rng=next_rng
    )
    
    train_step_fn = instantiate(cfg.loss, sde=sde, model=score_model, optimizer=optimiser)
    train_step_fn = jax.jit(train_step_fn)

    log.info("Stage : Training")

    for i in range(cfg.steps):
        batch = {'data': next(dataset)}
        rng, next_rng = jax.random.split(rng)
        (rng, train_state), loss = train_step_fn((next_rng, train_state), batch)
        if i % 50 == 0:
            print(i, ': ', loss)

    # log.info("Stage : Testing")

    x0 = next(dataset)
    ## p_0 (backward)
    t = cfg.eps
    sampler = jax.jit(get_pc_sampler(sde, score_model, (cfg.batch_size,), predictor=EulerMaruyamaManifoldPredictor, corrector=None, continuous=True, forward=False, eps=cfg.eps))
    rng, next_rng = jax.random.split(rng)
    x, _ = sampler(next_rng, train_state, t=t)
    likelihood_fn = get_likelihood_fn(sde, score_model, hutchinson_type='None', bits_per_dimension=False, eps=cfg.eps)
    logp, z, nfe = likelihood_fn(rng, train_state, x)
    print(nfe)
    prob = jnp.exp(logp)
    Path('logs/images').mkdir(parents=True, exist_ok=True)  # Create logs dir
    plot_and_save(None, x, prob, None, out=f"logs/images/x0_backw.jpg")
    prob = jnp.exp(dataset.log_prob(x0)) if hasattr(dataset, 'log_prob') else None
    plot_and_save(None, x0, prob, None, out=f"logs/images/x0_true.jpg")