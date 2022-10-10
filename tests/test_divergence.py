import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
from functools import partial
import math
import hydra
from hydra.utils import instantiate, get_class

import jax
from jax import numpy as jnp
import numpy as np
import haiku as hk

from score_sde.utils import TrainState, save, restore
from score_sde.sampling import get_pc_sampler
from riemannian_score_sde.sampling import EulerMaruyamaManifoldPredictor
from score_sde.utils import batch_mul


@hydra.main(config_path="../config", config_name="main")
def main(cfg):

    rng = jax.random.PRNGKey(cfg.seed)
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    sde = instantiate(cfg.flow, manifold=model_manifold)

    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng, manifold=data_manifold)
    x, z = next(dataset)

    def score_model(x, t, div=False, hutchinson_type="None"):
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=model_manifold,
        )
        if not div:
            return score(x, t)
        elif div == "divE":
            return score.divE(x, t, hutchinson_type)
        elif div == "div_split":
            return score.div_split(x, t, hutchinson_type)
        else:
            raise ValueError(f"{div}")

    score_model = hk.transform_with_state(score_model)

    rng, next_rng = jax.random.split(rng)
    params, state = score_model.init(rng=next_rng, x=x, t=jnp.zeros((x.shape[0], 1)))

    rng, step_rng = jax.random.split(rng)
    t = jax.random.uniform(step_rng, (x.shape[0],), minval=cfg.eps, maxval=sde.tf)
    t = t.reshape((x.shape[0], 1))
    assert data_manifold.belongs(x, atol=1e-6).all()
    out, _ = score_model.apply(params, state, next_rng, x=x, t=t)
    out = out.reshape(x.shape)
    assert model_manifold.is_tangent(out, x, atol=1e-6).all()
    # print("\n############ Hutchinson estimator ############")
    print("\n############ Score network ############")
    print("----------------data----------------")
    print(x[:5])
    print("--------------divE--------------")
    out, _ = score_model.apply(
        params, state, next_rng, x=x, t=t, div="divE", hutchinson_type="None"
    )
    print(out[:5])
    print("--------------divM--------------")
    out, _ = score_model.apply(
        params, state, next_rng, x=x, t=t, div="div_split", hutchinson_type="None"
    )
    print(out[:5])

    #
    # for div in ["divE", "div_split"]:
    #     try:
    #         print(f"----------------{div}----------------")
    #         M = 10000
    #         divs = jnp.zeros((M, x.shape[0]))

    #         @jax.jit
    #         def body(step, val):
    #             rng, divs = val
    #             rng, next_rng = jax.random.split(rng)
    #             out, _ = score_model.apply(
    #                 params,
    #                 state,
    #                 next_rng,
    #                 x=x,
    #                 t=t,
    #                 div=div,
    #                 hutchinson_type="Rademacher",
    #             )
    #             divs = divs.at[step, :].set(out)
    #             return rng, divs

    #         _, divs = jax.lax.fori_loop(0, M, body, (rng, divs))
    #         print("----------------mean----------------")
    #         print(jnp.mean(divs, 0)[:5])
    #         print("----------------std----------------")
    #         print(jnp.std(divs, 0)[:5])
    #         print("average variance", jnp.square(jnp.std(divs, 0)).mean())
    #     except:
    #         pass

    # rng, step_rng = jax.random.split(rng)
    # score = lambda x, t: score_model.apply(params, state, next_rng, x=x, t=t, div=False)[
    #     0
    # ]
    # div = lambda x, t, hutchinson_type: score_model.apply(
    #     params, state, next_rng, x=x, t=t, div="divE", hutchinson_type=hutchinson_type
    # )[0]

    # print("\n############ Vector basis ############")

    # def score_model(x, t, div=False, hutchinson_type="None"):
    #     output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
    #     score = instantiate(
    #         cfg.generator,
    #         cfg.architecture,
    #         cfg.embedding,
    #         output_shape,
    #         manifold=model_manifold,
    #     )

    #     def weights(x, t):
    #         if len(x.shape) == 1:
    #             out = jnp.zeros((output_shape))
    #             out = out.at[..., 0].set(1)
    #         else:
    #             out = jnp.zeros((x.shape[0], output_shape))
    #             out = out.at[..., 0].set(jnp.ones(x.shape[0]))
    #         return out

    #     score._weights = weights

    #     if not div:
    #         return score(x, t)
    #     elif div == "divE":
    #         return score.divE(x, t, hutchinson_type)
    #     elif div == "div_split":
    #         return score.div_split(x, t, hutchinson_type)
    #     else:
    #         raise ValueError(f"{div}")

    # score_model = hk.transform_with_state(score_model)

    # print("--------------divE--------------")
    # out, _ = score_model.apply(
    #     params, state, next_rng, x=x, t=t, div="divE", hutchinson_type="None"
    # )
    # print(out[:5])
    # print("--------------divM--------------")
    # out, _ = score_model.apply(
    #     params, state, next_rng, x=x, t=t, div="div_split", hutchinson_type="None"
    # )
    # print(out[:5])

    # print("\n############ Divergence Theorem ############")
    # radius = 1.0
    # N = 500
    # eps = 0.0
    # t = jax.random.uniform(step_rng, (1,), minval=cfg.eps, maxval=sde.tf)

    # theta = jnp.expand_dims(jnp.linspace(0, 2 * math.pi, N), -1)
    # xs = jnp.concatenate(
    #     [jnp.zeros((N, 1)), radius * jnp.cos(theta), radius * jnp.sin(theta)], axis=-1
    # )
    # n = jnp.concatenate(
    #     [-jnp.ones((N, 1)), jnp.zeros((N, 1)), jnp.zeros((N, 1))], axis=-1
    # )
    # v = score(xs, jnp.repeat(t, xs.shape[0], 0))
    # assert data_manifold.is_tangent(n, xs, atol=1e-6).all()
    # assert data_manifold.is_tangent(v, xs, atol=1e-6).all()
    # surface = 2 * math.pi * radius
    # rhs = data_manifold.embedding_metric.inner_product(v, n, xs) * surface
    # print("rhs", rhs.mean().item())

    # theta = jnp.linspace(eps, jnp.pi - eps, N // 2)
    # phi = jnp.linspace(-jnp.pi / 2 + eps, jnp.pi / 2 - eps, N)
    # theta, phi = jnp.meshgrid(theta, phi)
    # theta = theta.reshape(-1, 1)
    # phi = phi.reshape(-1, 1)
    # xs = jnp.concatenate(
    #     [jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)],
    #     axis=-1,
    # )
    # # assert (xs[..., 0] >= 0.).all()

    # div_v = div(xs, jnp.repeat(t, xs.shape[0], 0), "None")
    # volume = math.pi * math.pi
    # lambda_x = jnp.sin(theta)
    # lhs = batch_mul(div_v, lambda_x) * volume
    # print("lhs", lhs.mean().item())


if __name__ == "__main__":
    main()
