# %%
%load_ext autoreload
%autoreload 2

# %%
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import jax
import jax.numpy as jnp
from geomstats.geometry.hypersphere import Hypersphere
from riemannian_score_sde.datasets import Wrapped
from riemannian_score_sde.utils.vis import latlon_from_cartesian, cartesian_from_latlong
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

with open("/data/ziz/not-backed-up/mhutchin/score-sde/scripts/blender/mesh_utils.py") as file:
    exec(file.read())
# %%
sphere = Hypersphere(dim=2)
dataset = Wrapped(8, 'random', 5, (10000,), sphere, 0, False, 'unif')

samples = next(dataset)[0]
samples_ = latlon_from_cartesian(samples)

xx, yy = jnp.meshgrid(
    jnp.linspace(-jnp.pi/2, jnp.pi/2, 100)[1:-1], jnp.linspace(-jnp.pi, jnp.pi, 100)[1:-1]
)

coords = jnp.stack([xx, yy], axis=-1)
coords_ = cartesian_from_latlong(coords)
coords = latlon_from_cartesian(coords_)
xx, yy = coords[..., 0], coords[..., 1]

prob = dataset.log_prob(coords_)

plt.contour(xx, yy, prob)
plt.scatter(samples_[...,0], samples_[...,1], s=0.1)
plt.scatter(latlon_from_cartesian(dataset.mean)[..., 0], latlon_from_cartesian(dataset.mean)[..., 1], c='r')
plt.gca().set_aspect('equal')

# %%
def to_extrinsic(M):
    phi = M[..., 0]
    theta = M[..., 1]

    return jnp.stack(
        [
            jnp.sin(phi) * jnp.cos(theta),
            jnp.sin(phi) * jnp.sin(theta),
            jnp.cos(phi),
        ],
        axis=-1,
    )
        
import numpy as jnp
num_points = 30
phi = jnp.linspace(0, jnp.pi, num_points)
theta = jnp.linspace(0, 2 * jnp.pi, 2 * num_points + 1)[1:]
phi, theta = jnp.meshgrid(phi, theta, indexing="ij")
phi = phi.flatten()
theta = theta.flatten()
m = jnp.stack(
    [phi, theta], axis=-1
)  ### NOTE this ordering, I can change it but it'll be a pain, its latitude, longitude
density = 20
m_dense = jnp.stack(
    jnp.meshgrid(
        np.linspace(0, jnp.pi, density * num_points),
        np.linspace(0, 2 * jnp.pi, 2 * density * num_points + 1)[1:],
        indexing="ij",
    ),
    axis=-1,
)
x = to_extrinsic(m)
x_dense = to_extrinsic(m_dense)

mesh_obj = regular_square_mesh_to_obj(
    x.reshape((num_points, 2*num_points, 3)), wrap_y=True
)

save_obj(mesh_obj, "sphere.obj")

# %%

make_scalar_texture(
    lambda x: jnp.exp(dataset.log_prob(x)), 
    x_dense,
    f"data/wrap.png",
    cmap=sns.cubehelix_palette(
        light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
    ),
)

make_scalar_texture(
    lambda x: x[..., 0], 
    m_dense,
    f"data/phi.png",
    cmap=sns.cubehelix_palette(
        light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
    ),
)

make_scalar_texture(
    lambda x: x[..., 1], 
    m_dense,
    f"data/theta.png",
    cmap=sns.cubehelix_palette(
        light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
    ),
)

# %%

## Testing model loss
import os
import socket
import logging
from timeit import default_timer as timer
from tqdm import tqdm
from functools import partial

import jax
from jax import numpy as jnp
import optax
import haiku as hk

from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class, call

from score_sde.models.flow import SDEPushForward
from score_sde.losses import get_ema_loss_step_fn
from score_sde.utils import TrainState, save, restore
from score_sde.utils.loggers_pl import LoggerCollection
from score_sde.datasets import random_split, DataLoader, TensorDataset
from riemannian_score_sde.utils.normalization import compute_normalization
from riemannian_score_sde.utils.vis import plot, plot_ref

# %%
def build_model_ll_sampler_fn(
    run_path, **kwargs
):
    cfg = OmegaConf.load(run_path + "/.hydra/config.yaml")

    ckpt_path = os.path.join(run_path, cfg.ckpt_dir)

    rng = jax.random.PRNGKey(cfg.seed)
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    beta_schedule = instantiate(cfg.beta_schedule)
    flow = instantiate(cfg.flow, manifold=model_manifold, beta_schedule=beta_schedule)
    base = instantiate(cfg.base, model_manifold, flow)
    pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng)

    if isinstance(dataset, TensorDataset):
        # split and wrapp dataset into dataloaders
        train_ds, eval_ds, test_ds = random_split(
            dataset, lengths=cfg.splits, rng=next_rng
        )
        train_ds, eval_ds, test_ds = (
            DataLoader(train_ds, batch_dims=cfg.batch_size, rng=next_rng, shuffle=True),
            DataLoader(eval_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
            DataLoader(test_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
        )
    else:
        train_ds, eval_ds, test_ds = dataset, dataset, dataset

    def model(y, t, context=None):
        """Vector field s_\theta: y, t, context -> T_y M"""
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=model_manifold,
        )
        # TODO: parse context into embedding map
        if context is not None:
            t_expanded = jnp.expand_dims(t.reshape(-1), -1)
            if context.shape[0] != y.shape[0]:
                context = jnp.repeat(jnp.expand_dims(context, 0), y.shape[0], 0)
            context = jnp.concatenate([t_expanded, context], axis=-1)
        else:
            context = t
        return score(y, context)

    model = hk.transform_with_state(model)

    rng, next_rng = jax.random.split(rng)
    t = jnp.zeros((cfg.batch_size, 1))
    data, context = next(train_ds)
    params, state = model.init(rng=next_rng, y=transform.inv(data), t=t, context=context)

    schedule_fn = instantiate(cfg.scheduler)
    optimiser = optax.chain(instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn))
    opt_state = optimiser.init(params)

    train_state = restore(ckpt_path)

    model_w_dicts = (model, train_state.params_ema, train_state.model_state)
    likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False, **kwargs)
    likelihood_fn = jax.jit(likelihood_fn)

    model_w_dicts = (model, train_state.params_ema, train_state.model_state)
    sampler_kwargs = dict(N=100, eps=cfg.eps, predictor="GRW")
    sampler = pushforward.get_sampler(model_w_dicts, train=False, **sampler_kwargs)

    return lambda x: likelihood_fn(x)[0], partial(sampler, rng=rng)


def build_datasets(
    run_path,
):
    cfg = OmegaConf.load(run_path + "/.hydra/config.yaml")

    ckpt_path = os.path.join(run_path, cfg.ckpt_dir)

    rng = jax.random.PRNGKey(cfg.seed)
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    beta_schedule = instantiate(cfg.beta_schedule)
    flow = instantiate(cfg.flow, manifold=model_manifold, beta_schedule=beta_schedule)
    base = instantiate(cfg.base, model_manifold, flow)
    pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng)

    if isinstance(dataset, TensorDataset):
        # split and wrapp dataset into dataloaders
        train_ds, eval_ds, test_ds = random_split(
            dataset, lengths=cfg.splits, rng=next_rng
        )
        train_ds, eval_ds, test_ds = (
            DataLoader(train_ds, batch_dims=cfg.batch_size, rng=next_rng, shuffle=True),
            DataLoader(eval_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
            DataLoader(test_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
        )
        log.info(
            f"Train size: {len(train_ds.dataset)}. Val size: {len(eval_ds.dataset)}. Test size: {len(test_ds.dataset)}"
        )
    else:
        train_ds, eval_ds, test_ds = dataset, dataset, dataset

    return train_ds, eval_ds, test_ds
# %%
flood_path = "/data/ziz/not-backed-up/mhutchin/score-sde/results/flood_new/beta_schedule.beta_0=0.001,beta_schedule.beta_f=5,eps=0.001,generator=ambient,loss=ism,model=rsgm,scheduler=r3cosine/0"

flood_ll, flood_sampler = build_model_ll_sampler_fn(flood_path, tf=1)

def batch_fn(fn, batch_size):
    def batched_fn(x):
        lls = []
        x = x.reshape((-1, batch_size, x.shape[-1]))
        for i in range(x.shape[0]):
            lls.append(fn(x[i]))

        return jnp.array(lls)
    return batched_fn

make_scalar_texture(
    batch_fn(lambda x: jnp.exp(flood_ll(x)), x_dense.shape[0]),
    jnp.flip(jnp.flip(jnp.swapaxes(x_dense, 0,1),axis=1), axis=0),
    f"data/log_prob.png",
    cmap=sns.cubehelix_palette(
        light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
    ),
    vmin=0, vmax=3.0,
)
# make_scalar_texture(
#     lambda x: jnp.exp(flood_ll(x)),
#     jnp.flip(jnp.flip(jnp.swapaxes(x_dense, 0,1),axis=1), axis=0),
#     f"data/log_prob.png",
#     cmap=sns.cubehelix_palette(
#         light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
#     ),
#     vmin=0, vmax=3.0
# )

# %%
