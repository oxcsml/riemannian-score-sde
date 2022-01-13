import hydra
from hydra.utils import instantiate, get_class
import logging

import jax
from jax import numpy as jnp
import numpy as np
import haiku as hk
import optax

from score_sde.utils import TrainState, save, restore
# from score_sde.sampling import EulerMaruyamaManifoldPredictor, get_pc_sampler
# from score_sde.likelihood import get_likelihood_fn, get_pmap_likelihood_fn


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

    def score_model(x, t, div=False):
        score = instantiate(cfg.generator, cfg.architecture, output_shape, manifold=manifold)
        if not div:
            return score(x, t)
        else:
            return score.div(x, t)

    score_model = hk.transform_with_state(score_model)

    rng, next_rng = jax.random.split(rng)
    params, state = score_model.init(rng=next_rng, x=x, t=0)
    out = score_model.apply(params, state, next_rng, x=x, t=0)
    print(out)
    out = score_model.apply(params, state, next_rng, x=x, t=0, div=True)
    print(out)
    raise

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
        if i % 10 == 0:
            print(i, ': ', loss)

    # log.info("Stage : Testing")