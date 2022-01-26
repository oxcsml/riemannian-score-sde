import os
os.environ['GEOMSTATS_BACKEND'] = 'jax'
from functools import partial
import math
import hydra
from hydra.utils import instantiate, get_class

import jax
from jax import numpy as jnp
import numpy as np
import haiku as hk

import geomstats.backend as gs
from score_sde.likelihood import get_likelihood_fn
from score_sde.models import get_score_fn
from score_sde.utils.tmp import compute_normalization


@hydra.main(config_path="../config", config_name="main")
def main(cfg):

    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    sde = instantiate(cfg.sde, manifold=model_manifold)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng, manifold=data_manifold)
    z = transform.inv(next(dataset))


    def score_model(x, t, div=False, hutchinson_type='None'):
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(cfg.generator, cfg.architecture, output_shape, manifold=model_manifold)
        if not div:
            return score(x, t)
        else:
            return score.div(x, t, hutchinson_type)

    score_model = hk.transform_with_state(score_model)

    rng, next_rng = jax.random.split(rng)
    params, state = score_model.init(rng=next_rng, x=z, t=0)
    score_fn = get_score_fn(
                sde,
                score_model,
                params,
                state,
                continuous=True,
            )
    likelihood_fn = get_likelihood_fn(
        sde,
        score_fn,
        hutchinson_type="None",
        bits_per_dimension=False,
        eps=cfg.eps,
    )

    Z = compute_normalization(likelihood_fn, transform, model_manifold)
    print("Z", Z)


if __name__ == "__main__":
    main()