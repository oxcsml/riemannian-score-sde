import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
from functools import partial
import hydra
from hydra.utils import instantiate, get_class

import jax
from jax import numpy as jnp
import haiku as hk

from riemannian_score_sde.utils.normalization import compute_normalization


@hydra.main(config_path="../config", config_name="main")
def main(cfg):

    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    flow = instantiate(cfg.flow, manifold=model_manifold)
    base = instantiate(cfg.base, model_manifold, flow)
    pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng, manifold=data_manifold)
    y = transform.inv(next(dataset)[0])

    def score_model(y, t, context=None):
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=model_manifold,
        )
        return score(y, t)

    score_model = hk.transform_with_state(score_model)

    rng, next_rng = jax.random.split(rng)
    params, state = score_model.init(rng=next_rng, y=y, t=jnp.zeros((y.shape[0], 1)))

    model_w_dicts = (score_model, params, state)
    likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)

    Z = compute_normalization(likelihood_fn, data_manifold, N=200)
    print(f"Z = {Z:.2f}")


if __name__ == "__main__":
    main()
