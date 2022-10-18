# %%
%load_ext autoreload
%autoreload 2
# %%
import os
os.environ["GEOMSTATS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from geomstats.geometry.hyperbolic import Hyperbolic, Hyperboloid, PoincareBall
from geomstats.geometry._hyperbolic import _Hyperbolic

to_ball = _Hyperbolic._extrinsic_to_ball_coordinates
to_extr = _Hyperbolic._ball_to_extrinsic_coordinates

def proj(h, x):
    return to_ball(x) if isinstance(h, Hyperboloid) else x

def lift(h, x):
    return to_extr(x) if isinstance(h, PoincareBall) else x

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

rng = jax.random.PRNGKey(0)

def make_circle(ax=None):
    theta = jnp.linspace(0, 2*jnp.pi, 100)
    if ax is not None:
        # ax.plot(jnp.sin(theta), jnp.cos(theta), color='black')
        ax.add_patch(Circle((0, 0), 1., color="black", fill=False, linewidth=2, zorder=10))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
    else:
        plt.plot(jnp.sin(theta), jnp.cos(theta), color='black')
        # plt.add_patch(Circle((0, 0), 1., color="black", fill=False, linewidth=2, zorder=10))
        plt.gca().set_aspect('equal')
        plt.axis('off')

from riemannian_score_sde.utils.vis import make_disk_grid

def disk_plot(ax, h, prob_fn, N=150):
    cmap = sns.cubehelix_palette(light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True)
    make_circle(ax)

    xs, volume, lam = make_disk_grid(N)

    prob = jnp.exp(prob_fn(lift(h, xs)))# * mask 
    prob = prob.at[jnp.isnan(prob)].set(0.0)
    measure = prob * lam# ** 2 * mask
    Z = (measure * volume).mean()
    # Z = measure.mean() #TODO: should it be multiplied by the volume?
    print("volume = {:.2f}".format(Z.item()))

    xs = xs.reshape(N, N, 2)
    measure = measure.reshape(N, N)
    ax.pcolormesh(xs[:,  :, 0], xs[:,  :, 1], measure, cmap=cmap, linewidth=0, rasterized=True, shading="gouraud")

%matplotlib inline

rng = jax.random.PRNGKey(0)

with open("../blender/mesh_utils.py") as file:
    exec(file.read())

# %%

ball = Hyperbolic(dim=2, default_coords_type='ball')
ext = Hyperbolic(dim=2, default_coords_type='extrinsic')

# %%
from score_sde.sde import SDE
from score_sde.schedule import LinearBetaSchedule
from riemannian_score_sde.sde import NonCompactWrapNorm

beta_schedule = LinearBetaSchedule(beta_0=0.01, beta_f=5.0, tf=1)
sde = NonCompactWrapNorm(beta_schedule, ball, ref_scale=0.5, ref_mean=jnp.array([0.0, 0.0]), N=10000)

lim=1.5
points=25
line = jnp.linspace(-lim, lim, points)
tangent_grid = jnp.stack(jnp.meshgrid(line, line), axis=-1)
grid = ball.exp(tangent_grid, jnp.zeros_like(tangent_grid))
grid_ = grid.reshape(-1, 2)

plt.scatter(grid_[..., 0], grid_[..., 1], c=jnp.exp(sde.limiting_distribution_logp(grid_)))
make_circle()
plt.show()

# %%
lim=25/15
points=24
line = jnp.linspace(-lim, lim, points)
grid = jnp.stack(jnp.meshgrid(line, line), axis=-1)

z = (grid[..., 0]**2 - grid[..., 1]**2)

hyperbolic_paraboloid_grid = jnp.concatenate([grid, z[..., None]], axis=-1)

mesh_obj = regular_square_mesh_to_obj(
    hyperbolic_paraboloid_grid, wrap_x=False, wrap_y=False
)
save_obj(mesh_obj, "data/hyperbolic_paraboloid.obj")

# %%
from riemannian_score_sde.datasets.simple import WrapNormMixtureDistribution

dense_line = jnp.linspace(-lim, lim, 10*points)
dense_grid = np.stack(jnp.meshgrid(dense_line, dense_line), axis=-1)
z = jnp.sqrt(1 + dense_grid[...,0]**2 + dense_grid[...,1]**2)
paraboloid_grid = jnp.concatenate([z[..., None], dense_grid], axis=-1)
ball_grid = ball.change_coordinates_system(paraboloid_grid, 'extrinsic', 'ball')


from riemannian_score_sde.datasets.simple import WrapNormMixtureDistribution
mean = lift(ball, 0.4*jnp.array([[-1, 0.0],[1, 0.0],[0.0, -1],[0.0, 1]]))
scale = jnp.array([[0., 0.15, 0.5],[0., 0.15, 0.5],[0., 0.5, 0.15],[0., 0.5, 0.15]])
ref_dist = WrapNormMixtureDistribution(10000, ext, mean, scale)


plt.scatter(ball_grid[..., 0], ball_grid[..., 1], c=jnp.exp(ref_dist.log_prob(to_extr(ball_grid))))
theta = jnp.linspace(0, 2*jnp.pi, 100)
plt.plot(jnp.sin(theta), jnp.cos(theta))
plt.gca().set_aspect('equal')

ref_ll = jnp.exp(ref_dist.log_prob(paraboloid_grid.reshape((-1, 3))))

make_scalar_texture(
    lambda x: jnp.exp(ref_dist.log_prob(x)),
    paraboloid_grid,
    "data/ref.png",
    # cmap='viridis',
    cmap=sns.cubehelix_palette(
        light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
    ),
    vmax=ref_ll.max()
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
    likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)
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


def make_model_texture(
    ll_fn,
    paraboloid_grid,
    save_name,
    **kwargs
):
    make_scalar_texture(
        lambda x: jnp.exp(ll_fn(x)),
        paraboloid_grid,
        f"data/{save_name}.png",
        # cmap='viridis',
        cmap=sns.cubehelix_palette(
            light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
        ),
        **kwargs
    )

# %%    

ds, _, _ = build_datasets(    
    "/data/ziz/not-backed-up/mhutchin/score-sde/results/hyp_closer/beta_schedule=quadratic,steps=100000/0"
)
# %%

quadratic_ll, quadratic_sampler = build_model_ll_sampler_fn(
    "/data/ziz/not-backed-up/mhutchin/score-sde/results/hyp_closer/beta_schedule.beta_f=2,beta_schedule=quadratic,steps=100000/0"
)
# quadratic_ll = ll(paraboloid_grid.reshape((-1, 3)))
make_model_texture(
    quadratic_ll, 
    paraboloid_grid,
    "rsgm_quadratic",
    # vmax=ref_ll.max()
)
# %%    
make_model_texture(
    build_model_ll_fn("/data/ziz/not-backed-up/mhutchin/score-sde/results/hyp_closer/steps=100000/0"), 
    paraboloid_grid,
    "rsgm",
    # vmax=ref_ll.max()
)
# %%    
make_model_texture(
    build_model_ll_fn("/data/ziz/not-backed-up/mhutchin/score-sde/results/hyp_closer/beta_schedule=quadratic,eps=1e-4,optim.learning_rate=5e-4,steps=100000/0"), 
    paraboloid_grid,
    "rsgm_quadratic_eps",
    # vmax=ref_ll.max()
)
# %% 

exp_wrap_ll, exp_wrap_sampler = build_model_ll_sampler_fn(
    "/data/ziz/not-backed-up/mhutchin/score-sde/results/exp_hyp/beta_schedule.beta_f=5,steps=100000/0", 
)
# quadratic_ll = ll(paraboloid_grid.reshape((-1, 3)))
make_model_texture(
    exp_wrap_ll, 
    paraboloid_grid,
    "exp_wrap",
    # vmax=ref_ll.max()
)
# %%
make_model_texture(
    build_model_ll_fn("/data/ziz/not-backed-up/mhutchin/score-sde/results/hyp_closer/beta_schedule.beta_f=2,beta_schedule=quadratic,steps=100000/0"), 
    paraboloid_grid,
    "rsgm_quadratic_2",
    # vmax=ref_ll.max()
)
# %%

plt.scatter(ball_grid[..., 0], ball_grid[..., 1], c=jnp.exp(build_model_ll_fn("/data/ziz/not-backed-up/mhutchin/score-sde/results/hyp_closer/beta_schedule.beta_f=2,beta_schedule=quadratic,steps=100000/0")(to_extr(ball_grid.reshape((-1, 2)))).reshape(ball_grid.shape[:2])))
theta = jnp.linspace(0, 2*jnp.pi, 100)
plt.plot(jnp.sin(theta), jnp.cos(theta))
plt.gca().set_aspect('equal')



# %%
h=Hyperbolic(dim=2, default_coords_type='extrinsic')
ref_dist = WrapNormMixtureDistribution(
    1000, h, mean=h.identity[None, :], scale=0.5)

make_model_texture(
    ref_dist.log_prob,
    paraboloid_grid,
    'base_dist',
)
# %%
from riemannian_score_sde.datasets.simple import WrapNormMixtureDistribution
mean = lift(ball, jnp.array([[0.0, 0.0]]))
scale = jnp.array([[0., 0.5, 0.5]])
base_dist = WrapNormMixtureDistribution(10000, ext, mean, scale)


plt.scatter(ball_grid[..., 0], ball_grid[..., 1], c=jnp.exp(base_dist.log_prob(to_extr(ball_grid))))
theta = jnp.linspace(0, 2*jnp.pi, 100)
plt.plot(jnp.sin(theta), jnp.cos(theta))
plt.gca().set_aspect('equal')

base_ll = jnp.exp(base_dist.log_prob(paraboloid_grid.reshape((-1, 3))))

make_scalar_texture(
    lambda x: jnp.exp(base_dist.log_prob(x)),
    paraboloid_grid,
    "data/base.png",
    cmap='viridis',
    vmax=ref_ll.max()
)
# %%
from riemannian_score_sde.utils.vis import plot_ref
# %%
samples = quadratic_sampler(shape=(10000,), context=None)
log_prob = quadratic_ll(samples)

cmap = sns.cubehelix_palette(
    light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
)

xt = to_ball(samples)
c = jnp.exp(log_prob)

plt.scatter(
    xt[..., 0], xt[..., 1], alpha=0.3, s=2, c=c, label="model", cmap=cmap
)
make_circle()
# %%
samples = next(ds)[0]
log_prob = ds.log_prob(samples)

cmap = sns.cubehelix_palette(
    light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
)

xt = to_ball(samples)
c = jnp.exp(log_prob)

plt.scatter(
    xt[..., 0], xt[..., 1], alpha=0.3, s=2, c=c, label="model", cmap=cmap
)
make_circle()
# %%
