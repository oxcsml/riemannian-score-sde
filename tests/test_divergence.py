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

from score_sde.utils import TrainState, save, restore
from score_sde.sampling import EulerMaruyamaManifoldPredictor, get_pc_sampler
from score_sde.likelihood import get_likelihood_fn


@hydra.main(config_path="../config", config_name="main")
def main(cfg):

    rng = jax.random.PRNGKey(cfg.seed)
    manifold = instantiate(cfg.manifold)
    sde = instantiate(cfg.sde, manifold)

    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng, manifold=manifold)
    x = next(dataset)

    output_shape = get_class(cfg.generator._target_).output_shape(manifold)

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

    rng, step_rng = jax.random.split(rng)
    x = jax.random.normal(step_rng, x.shape)
    x = x / jnp.linalg.norm(x)
    t = jax.random.uniform(step_rng, (x.shape[0],), minval=cfg.eps, maxval=sde.T)
    M = 1000
    out, _ = score_model.apply(params, state, next_rng, x=x, t=t, div=True, hutchinson_type='None')
    print(x[:5])
    print(out.shape)
    print(out[:5])

    divs = jnp.zeros((M, x.shape[0]))
    @jax.jit
    def body(step, val):
        rng, divs = val
        rng, next_rng = jax.random.split(rng)
        out, _ = score_model.apply(params, state, next_rng, x=x, t=t, div=True, hutchinson_type='Rademacher')
        divs = divs.at[step, :].set(out)
        return rng, divs
    _, divs = jax.lax.fori_loop(0, M, body, (rng, divs))
    print(jnp.mean(divs, 0)[:5])
    print(jnp.std(divs, 0)[:5])

    rng, step_rng = jax.random.split(rng)
    score = lambda x, t: score_model.apply(params, state, next_rng, x=x, t=t, div=False)[0]
    div = lambda x, t, hutchinson_type: score_model.apply(params, state, next_rng, x=x, t=t, div=True, hutchinson_type=hutchinson_type)[0]


    radius = 1.
    N = 200
    eps = 1e-3
    t = jax.random.uniform(step_rng, (1,), minval=cfg.eps, maxval=sde.T)

    theta = jnp.expand_dims(jnp.linspace(0, 2 * math.pi, N), -1)
    xs = jnp.concatenate([jnp.zeros((N, 1)), radius * jnp.cos(theta), radius * jnp.sin(theta)], axis=-1)
    n = jnp.concatenate([-jnp.ones((N, 1)), jnp.zeros((N, 1)), jnp.zeros((N, 1))], axis=-1)
    v = score(xs, jnp.repeat(t, xs.shape[0], 0))
    assert manifold.is_tangent(n, xs, atol=1e-6).all()
    assert manifold.is_tangent(v, xs, atol=1e-6).all()
    surface = 2 * math.pi * radius
    rhs = manifold.embedding_metric.inner_product(v, n, xs) * surface
    print("rhs", rhs.mean().item())

    theta = jnp.linspace(eps, jnp.pi - eps, N // 2)
    phi = jnp.linspace(-jnp.pi / 2 + eps, jnp.pi / 2 - eps, N)
    theta, phi = jnp.meshgrid(theta, phi)
    theta = theta.reshape(-1, 1)
    phi = phi.reshape(-1, 1)
    xs = jnp.concatenate([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi), 
        jnp.cos(theta)
    ], axis=-1)
    assert (xs[..., 0] >= 0.).all()

    div_v = div(xs, jnp.repeat(t, xs.shape[0], 0), 'None')
    # volume = math.pi * math.pi
    volume = 2 * math.pi * radius ** 2
    lambda_x = jnp.sin(theta)
    # lambda_x = jnp.ones_like(theta)
    lhs = div_v * lambda_x * volume
    print("lhs", lhs.mean().item())

if __name__ == "__main__":
    main()