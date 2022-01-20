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
from score_sde.models import get_score_fn

log = logging.getLogger(__name__)


def run(cfg):
    log.info("Stage : Startup")

    rng = jax.random.PRNGKey(cfg.seed)
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    sde = instantiate(cfg.sde, manifold=model_manifold)

    log.info("Stage : Instantiate dataset")

    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng, manifold=data_manifold)
    x = transform.inv(next(dataset))

    log.info("Stage : Instantiate model")

    def score_model(x, t):
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator, cfg.architecture, output_shape, manifold=model_manifold
        )
        return score(x, t)

    score_model = hk.transform_with_state(score_model)

    rng, next_rng = jax.random.split(rng)
    params, state = score_model.init(rng=next_rng, x=x, t=0)

    log.info("Stage : Instantiate optimiser")

    schedule_fn = instantiate(cfg.scheduler)
    optimiser = optax.chain(
        instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn)
    )
    opt_state = optimiser.init(params)

    rng, next_rng = jax.random.split(rng)
    train_state = TrainState(
        opt_state=opt_state,
        model_state=state,
        step=0,
        params=params,
        ema_rate=cfg.ema_rate,
        params_ema=params,
        rng=next_rng,
    )

    train_step_fn = instantiate(
        cfg.loss, sde=sde, model=score_model, optimizer=optimiser
    )

    train_step_fn = jax.jit(train_step_fn)

    log.info("Stage : Training")

    for i in range(cfg.steps):
        batch = {"data": transform.inv(next(dataset))}
        rng, next_rng = jax.random.split(rng)
        (rng, train_state), loss = train_step_fn((next_rng, train_state), batch)
        if i % 50 == 0:
            print(f"{i:4d}: {loss:.3f}")

    log.info("Stage : Testing")

    x0 = next(dataset)
    ## p_0 (backward)
    t = cfg.eps
    # sampler = jax.jit(
    sampler = get_pc_sampler(
        sde.reverse(
            get_score_fn(
                sde, score_model, train_state.params_ema, train_state.model_state
            )
        ),
        100,
        predictor="EulerMaruyamaManifoldPredictor",
        corrector=None,
        eps=cfg.eps,
    )
    # )
    # sampler = get_pc_sampler(
    #     sde.reverse(
    #         get_score_fn(
    #             sde, score_model, train_state.params_ema, train_state.model_state
    #         )
    #     ),
    #     1000,
    #     predictor="EulerMaruyamaManifoldPredictor",
    #     corrector=None,
    #     eps=cfg.eps,
    # )
    rng, next_rng = jax.random.split(rng)
    x, _ = sampler(next_rng, sde.sample_limiting_distribution(rng, x0.shape))
    y = transform(x)
    log.info("Jitting likelihood")
    # likelihood_fn = jax.jit(
    #     get_likelihood_fn(
    #         sde,
    #         get_score_fn(
    #             sde, score_model, train_state.params_ema, train_state.model_state
    #         ),
    #         hutchinson_type="None",
    #         bits_per_dimension=False,
    #         eps=cfg.eps,
    #         N=100,
    #     )
    # )
    # likelihood_fn = get_likelihood_fn(
    #     sde,
    #     get_score_fn(sde, score_model, train_state.params_ema, train_state.model_state),
    #     hutchinson_type="None",
    #     bits_per_dimension=False,
    #     eps=cfg.eps,
    # )
    likelihood_fn = get_likelihood_fn(
        sde,
        get_score_fn(
            sde,
            score_model,
            train_state.params_ema,
            train_state.model_state,
            continuous=True,
        ),
        hutchinson_type="None",
        bits_per_dimension=False,
        eps=cfg.eps,
    )
    # TODO: take into account logdetjac of transform
    log.info("Running likelihood")
    logp, z, nfe = likelihood_fn(rng, transform.inv(y))
    print(logp)
    print(nfe)
    logp -= transform.log_abs_det_jacobian(x, y)
    prob = jnp.exp(logp)
    print(prob)
    Path("logs/images").mkdir(parents=True, exist_ok=True)  # Create logs dir
    plot_and_save(None, y, prob, None, out=f"logs/images/x0_backw.jpg")
    prob = jnp.exp(dataset.log_prob(x0)) if hasattr(dataset, "log_prob") else None
    plot_and_save(None, x0, prob, None, out=f"logs/images/x0_true.jpg")
