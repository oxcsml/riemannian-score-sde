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


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


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

    def loglike(params, state, x):
        rng = jax.random.PRNGKey(cfg.seed)
        likelihood_fn = get_likelihood_fn(
            sde,
            get_score_fn(
                sde,
                score_model,
                params,
                state,
                continuous=True,
            ),
            hutchinson_type="None",
            bits_per_dimension=False,
            eps=cfg.eps,
        )

        z = transform.inv(x)
        logp, zT, nfe = likelihood_fn(rng, z)
        logp -= transform.log_abs_det_jacobian(z, x)
        return logp, zT, nfe

    rng, next_rng = jax.random.split(rng)
    params, state = score_model.init(rng=next_rng, x=z, t=0)

    N = 500
    eps = 0.
    theta = jnp.linspace(eps, jnp.pi - eps, N // 2)
    phi = jnp.linspace(eps, 2 * jnp.pi - eps, N)

    theta, phi = jnp.meshgrid(theta, phi)
    theta = theta.reshape(-1, 1)
    phi = phi.reshape(-1, 1)
    xs = jnp.concatenate([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi), 
        jnp.cos(theta)
    ], axis=-1)

    logp, zT, nfe = loglike(params, state, xs)
    print("nfe", nfe)
    belongs_manifold = model_manifold.belongs(zT, atol=gs.atol/2)
    print("belongs_manifold", jnp.sum(belongs_manifold) / belongs_manifold.shape[0])

    prob = jnp.exp(logp)
    volume = (2 * np.pi) * np.pi
    lambda_x = jnp.sin(theta).reshape((-1))
    Z = (prob * lambda_x).mean() * volume
    print("Z", Z.item())

if __name__ == "__main__":
    main()