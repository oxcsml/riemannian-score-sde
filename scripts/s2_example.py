from functools import partial

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
import geomstats.backend as gs

from score_sde.sampling import EulerMaruyamaManifoldPredictor, get_pc_sampler
from score_sde.sde import SDE, batch_mul, Brownian
from score_sde.utils import TrainState, replicate
from score_sde.models import MLP, ScoreFunctionWrapper, ScoreNetwork, Squash, Concat, ConcatSquash
from score_sde.losses import get_pmap_step_fn, get_step_fn
import jax
from jax import device_put
from jax import numpy as jnp
import haiku as hk
import optax
import numpy as np

from scripts.utils import plot_and_save, plot_and_save_video, vMF_pdf
from scripts.utils import save, restore


def remove_background(ax):
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return ax


def plot_and_save3(
    x0, xt, prob, grad, size=20, dpi=300, out="out.jpg", color="red"
):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax = remove_background(ax)
    fig.subplots_adjust(left=-0.2, bottom=-0.2, right=1.2, top=1.2, wspace=0, hspace=0)
    ax.view_init(elev=30, azim=45)
    cmap = sns.cubehelix_palette(as_cmap=True)
    sphere = visualization.Sphere()
    sphere.draw(ax, color="red", marker=".")
    # sphere.plot_heatmap(ax, pdf, n_points=16000, alpha=0.2, cmap=cmap)
    if x0 is not None:
        cax = ax.scatter(x0[:,0], x0[:,1], x0[:,2], s=200, color='green')
    if xt is not None:
        x, y, z = xt[:,0], xt[:,1], xt[:,2]
        c = prob if prob is not None else np.ones([*xt.shape[:-1]])
        cax = ax.scatter(x, y, z, s=50, c=c, cmap=cmap)
    if grad is not None:
        u, v, w = grad[:, 0], grad[:, 1], grad[:, 2]
        quiver = ax.quiver(x, y, z, u, v, w, length=.5, lw=2, normalize=False, cmap=cmap)
        quiver.set_array(c)

    plt.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_and_save_video3(
    traj_x, traj_f, traj_grad_f, timesteps, size=20, fps=10, dpi=100, out="out.mp4", color="red"
):
    """Render a set of geodesics and save it to an mpeg 4 file."""
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=fps)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax = remove_background(ax)
    fig.subplots_adjust(left=-0.2, bottom=-0.2, right=1.2, top=1.2, wspace=0, hspace=0)
    # fig.subplots_adjust(wspace=-100, hspace=100)
    ax.view_init(elev=30, azim=45)
    colours = sns.cubehelix_palette(n_colors=traj_x.shape[1])
    cmap = sns.cubehelix_palette(as_cmap=True)
    sphere = visualization.Sphere()
    sphere.draw(ax, color=color, marker=".")
    n = traj_x.shape[0]
    with writer.saving(fig, out, dpi=dpi):
        for i in range(1, n):
            text = ax.text(x=0.5, y=0.5, z=-1., s=timesteps[i], size=50, c='black')
            x, y, z = traj_x[i,:,0], traj_x[i,:,1], traj_x[i,:,2]
            c = traj_f[i]
            scatter = ax.scatter(x, y, z, s=50, c=c, cmap=cmap)
            # scatter = ax.scatter(x, y, z, s=50, c=colours)
            u, v, w = traj_grad_f[i,:,0], traj_grad_f[i,:,1], traj_grad_f[i,:,2]
            # quiver = ax.quiver(x, y, z, u, v, w, length=1., normalize=False)
            quiver = ax.quiver(x, y, z, u, v, w, length=.5, lw=2, normalize=False, cmap=cmap)
            quiver.set_array(c)
            # quiver.set_array(jnp.sqrt(jnp.sum(jnp.square(traj_grad_f[i, :, :]), -1)))
            # break
            writer.grab_frame()
            text.remove()
            scatter.remove()
            quiver.remove()
    # plt.savefig('test.png', dpi=dpi, transparent=False)
    # plt.close(fig)


class vMFDataset:
    def __init__(
        self, batch_dims, rng, manifold, mu=None, kappa=1.
    ):
        self.manifold = manifold
        self.mu = mu
        self.kappa = kappa
        self.batch_dims = batch_dims
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        # rng = jax.random.split(self.rng)

        samples = self.manifold.random_von_mises_fisher(
            mu=self.mu,
            kappa=self.kappa,
            n_samples=np.prod(self.batch_dims)
        )
        samples = samples.reshape([*self.batch_dims, samples.shape[-1]])

        return samples
        # return jnp.expand_dims(samples, axis=-1)


class DeltaDataset:
    def __init__(
        self, batch_dims, mu=None
    ):
        self.mu = mu
        self.batch_dims = batch_dims

    def __iter__(self):
        return self

    def __next__(self):
        samples = jnp.repeat(self.mu.reshape(1, -1), jnp.array(self.batch_dims), 0)
        return samples


def test_score():
    manifold = Hypersphere(dim=2)
    N = 1000
    batch_size = 256 * 1
    rng = jax.random.PRNGKey(0)
    dataset = DeltaDataset([batch_size], mu=jnp.array([1., 0., 0.]))
    sde = Brownian(manifold, T=5, N=N)

    data = next(dataset)
    rng, step_rng = jax.random.split(rng)
    t = jax.random.uniform(step_rng, (1,), minval=1e-3, maxval=sde.T)
    t = 0.05 + 1e-6
    print(t)
    rng, step_rng = jax.random.split(rng)
    perturbed_data = sde.marginal_sample(step_rng, data, t)
    t = jnp.ones(data.shape[0]) * t
    # score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

    logp_grad_fn = jax.value_and_grad(sde.marginal_log_prob, argnums=1, has_aux=False)
    logp, logp_grad = jax.vmap(logp_grad_fn)(data, perturbed_data, t)
    # logp = manifold.log_heat_kernel(data, perturbed_data, t)
    # p_grad = manifold.grad_heat_kernel(data, perturbed_data, t)
    # logp_grad = p_grad / jnp.exp(logp)
    
    logp_grad = manifold.to_tangent(logp_grad, perturbed_data)
    assert (jnp.sum(perturbed_data * logp_grad, axis=-1) < 1e-6).all()

    plot_and_save3(data, perturbed_data, jnp.exp(logp), logp_grad, out="images/s2_true_score.jpg")


@partial(jax.jit, static_argnums=(3))
def value_and_grad_log_marginal_prob(x0, x, t, manifold):
    def log_marginal_prob(x0, x, t):
        return jnp.reshape(manifold.log_heat_kernel(x0, x, t), ())

    logp_grad_fn = jax.value_and_grad(log_marginal_prob, argnums=1, has_aux=False)
    logp, logp_grad = jax.vmap(logp_grad_fn)(x0, x, jnp.ones((x.shape[0], 1)) * t)
    logp_grad = manifold.to_tangent(logp_grad, x)
    return logp, logp_grad


@partial(jax.jit, static_argnums=(4, 5, 6))
def EulerMaruyama(previous_x, traj, dt, timesteps, sde, manifold, forward=True):
    rng = jax.random.PRNGKey(0)
    x0 = jnp.repeat(gs.array([[1., 0., 0.]]), previous_x.shape[0], 0)
    N = len(timesteps)

    def body(step, val):
        rng, x, traj = val
        t = timesteps[step]
        t = jnp.broadcast_to(t, (x.shape[0], 1))
        sign = 1 if forward else -1
        rng, z = manifold.random_normal_tangent(state=rng, base_point=x, n_samples=x.shape[0])
        drift, diffusion = sde.sde(x, t)
        tangent_vector = sign * drift * dt + batch_mul(diffusion, jnp.sqrt(dt) * z)
        x = manifold.metric.exp(tangent_vec=tangent_vector, base_point=x)
        logp, logp_grad = value_and_grad_log_marginal_prob(x0, x, t, manifold)
        vector = diffusion ** 2 * logp_grad if forward else sign * drift

        traj = traj.at[step, :, :3].set(x)
        traj = traj.at[step, :, 3:6].set(vector)
        traj = traj.at[step, :, -1].set(logp)
        return rng, x, traj
    _, _, traj = jax.lax.fori_loop(0, N, body, (rng, previous_x, traj))
    return traj

   
# def forward_or_backward_process(forward=True):
def forward_or_backward_process(forward=False):
    N = 1000
    T = 5
    n_samples = 512
    S2 = Hypersphere(dim=2)
    sde = Brownian(S2, T=T, N=N)
    x0 = gs.array([[1., 0., 0.]])
    x0b = jnp.repeat(x0, n_samples, 0)
    eps = 1e-3
    dt = (T - eps) / N

    if forward:
        S2_EulerMaruyama = partial(EulerMaruyama, sde=sde, manifold=S2, forward=True)
        # previous_x = S2.random_von_mises_fisher(kappa=15, n_samples=n_samples)
        previous_x = next(DeltaDataset([n_samples], mu=x0))
        timesteps = jnp.linspace(eps, T, N)
    else:
        # params = restore('experiments', 'params')
        # state = restore('experiments', 'state')
        # score_model = hk.transform_with_state(partial(score_inv, manifold=S2))
        # score_fn = lambda x, t: score_model.apply(params, state, None, x=x, t=t)[0]
        score_fn = lambda x, t: value_and_grad_log_marginal_prob(x0b, x, t, S2)[1]
        rsde = sde.reverse(score_fn)
        S2_EulerMaruyama = partial(EulerMaruyama, sde=rsde, manifold=S2, forward=False)
        previous_x = Brownian(S2, T=T, N=N).prior_sampling(jax.random.PRNGKey(0), [n_samples])
        timesteps = jnp.linspace(T, eps, N)

    traj = jnp.zeros((N, previous_x.shape[0], (S2.dim + 1) * 2 + 1))
    traj = S2_EulerMaruyama(previous_x, traj, 2*dt, timesteps)

    idx = (jnp.arange(N) % 5) == 0
    traj_x = traj[idx, :, :3]
    traj_logp_grad = traj[idx, :, 3:6]
    traj_logp_grad = traj_logp_grad
    traj_logp = traj[idx, :, -1]
    timesteps = list(map(lambda x: f"{x:.2f}", timesteps[idx]))
    # pdf = partial(vMF_pdf, mu=x0, kappa=kappa)
    # plot_and_save_video(traj_x, pdf=None, out="forward.mp4")
    direction = 'forward' if forward else 'backward'
    plot_and_save_video3(traj_x, jnp.exp(traj_logp), traj_logp_grad, timesteps, size=20, fps=10, dpi=100, out=f"images/s2_{direction}.mp4",)


### Score networks ###

# score_model = lambda x, t: manifold.metric.log(x0, x)

def score_true(x, t, x0, manifold):
    def log_marginal_prob(x0, x, t):
        return jnp.reshape(manifold.log_heat_kernel(x0, x, t), ())

    logp_grad_fn = jax.value_and_grad(log_marginal_prob, argnums=1, has_aux=False)
    if isinstance(t, (int, float)):
        t = jnp.ones((x.shape[0], 1)) * t
    # x0_expanded = x0 if len(x0.shape) >= 2 else jnp.repeat(x0.reshape(1, -1), x.shape[0], 0)
    _, logp_grad = jax.vmap(logp_grad_fn)(x0, x, t)
    logp_grad = manifold.to_tangent(logp_grad, x)
    return logp_grad

def score_ambiant(x, t, manifold):
    return manifold.to_tangent(ScoreFunctionWrapper(MLP(hidden_shapes=3*[512], output_shape=3, act='sin'))(x, t), x)  # NOTE: regularize orthogonal component?

# @partial(jax.jit, static_argnums=(2))
def score_inv(x, t, manifold):
    """ output_shape = dim(Isom(manifold)) """
    invariant_basis = manifold.invariant_basis(x)
    # layer = MLP(hidden_shapes=3*[512], output_shape=3, act='sin')
    # weights = ScoreFunctionWrapper(layer)(x, t)
    weights = ScoreNetwork(output_shape=3)(x, t)
    # weights = ConcatSquash(output_shape=3, layer=layer)(x, t)  # output_shape = dim(Isom(manifold))
    return (gs.expand_dims(weights, -2) * invariant_basis).sum(-1)


def main():
    rng = jax.random.PRNGKey(0)
    N = 100
    T = 2
    manifold = Hypersphere(dim=2)
    sde = Brownian(manifold, T=T, N=N)
    batch_size = 256 * 2

    x0 = jnp.array([1., 0., 0.])
    # dataset_init = vMFDataset([batch_size], jax.random.PRNGKey(0), manifold, mu=x0, kappa=15)
    dataset_init = DeltaDataset([batch_size], mu=x0)
    x = next(dataset_init)

    # score_model = partial(score_true, manifold=manifold, x0=jnp.repeat(x0, x.shape[0], 0))
    # score_model = partial(score_ambiant, manifold=manifold)
    score_model = partial(score_inv, manifold=manifold)
    score_model = hk.transform_with_state(score_model)
    rng, next_rng = jax.random.split(rng)
    params, state = score_model.init(rng=next_rng, x=x, t=0)
    # print(score_model.apply(params, state, jax.random.PRNGKey(0), x=x, t=0)[0].shape)

    ### Optimiser ###

    warmup_steps, steps = 10, 100
    # warmup_steps, steps = 0, 20
    # warmup_steps, steps = 0, 1
    # steps = 1

    # lr=2e-4
    lr=5e-4

    schedule_fn = optax.join_schedules([
        optax.linear_schedule(init_value=0.0, end_value=1.0, transition_steps=warmup_steps),
        optax.cosine_decay_schedule(init_value=1.0, decay_steps = steps - warmup_steps, alpha=0.0),
    ], [warmup_steps])

    optimiser = optax.chain(
        optax.clip_by_global_norm(jnp.inf),
        optax.adam(lr, b1=.9, b2=0.999, eps=1e-8),
        optax.scale_by_schedule(schedule_fn)
    )
    opt_state = optimiser.init(params)

    rng, next_rng = jax.random.split(rng)
    train_state = TrainState(
        opt_state=opt_state, model_state=state, step=0, params=params, ema_rate=0.999, params_ema=params, rng=next_rng
    )
    # train_step_fn = get_pmap_step_fn(sde, score_model, optimiser, True, reduce_mean=False, continuous=True, likelihood_weighting=False)
    train_step_fn = get_step_fn(sde, score_model, optimiser, True, reduce_mean=False, continuous=True, likelihood_weighting=False, eps=1e-2)
    p_train_step = jax.pmap(partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
    p_train_state = replicate(train_state)

    ### Training ###

    # dataset = vMFDataset([1,1,batch_size], jax.random.PRNGKey(0), manifold, kappa=15)
    dataset = dataset_init
    for i in range(steps):
        batch = {
            'data': next(dataset),
            'label': None
        }

        # rng = p_train_state.rng[0]
        # rng = train_state.rng  # TODO: train_state.rng is not updated!

        # next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        # rng = next_rng[0]
        # next_rng = next_rng[1:]
        rng, next_rng = jax.random.split(rng)
        # (_, p_train_state), loss = p_train_step((next_rng, p_train_state), batch)
        (rng, train_state), loss = train_step_fn((next_rng, train_state), batch)
        # if i%100 == 0:
        print(i, ': ', loss)

    ### Analysis ###

    ## p_T
    x0 = next(dataset_init)
    # rng, next_rng = jax.random.split(rng)
    # x = sde.prior_sampling(next_rng, [batch_size])
    # score, _ = score_model.apply(train_state.params, train_state.model_state, next_rng, x=x, t=sde.T)
    # prob = jax.vmap(sde.marginal_log_prob)(x0, x, jnp.ones(x.shape[0]) * sde.T)
    # plot_and_save3(None, x, jnp.exp(prob), score, out="images/s2_xT_score.jpg")

    ## p_t (forward)
    t = 0.1
    # x = sde.marginal_sample(step_rng, x0, t)  # Equivalent to below
    # sampler = get_pc_sampler(sde, None, (batch_size,), predictor=EulerMaruyamaManifoldPredictor, corrector=None, continuous=True, forward=True, eps=1e-3)
    # rng, next_rng = jax.random.split(rng)
    # x, _ = sampler(next_rng, train_state, x=x0, t=t)
    
    # logp_grad_fn = jax.value_and_grad(sde.marginal_log_prob, argnums=1, has_aux=False)
    # logp, logp_grad = jax.vmap(logp_grad_fn)(x0, x, jnp.ones(x.shape[0]) * t)
    # logp_grad = sde.manifold.to_tangent(logp_grad, x)
    # plot_and_save3(x0, x, jnp.exp(logp), logp_grad, out="images/s2_xt_forw_score_true.jpg")
    # rng, next_rng = jax.random.split(rng)
    # score, _ = score_model.apply(train_state.params, train_state.model_state, next_rng, x=x, t=t)
    # prob = jax.vmap(sde.marginal_log_prob)(x0, x, jnp.ones(x.shape[0]) * t)
    # plot_and_save3(x0, x, jnp.exp(prob), score, out="images/s2_xt_forw_score.jpg")

    ## p_t (backward)
    sampler = get_pc_sampler(sde, score_model, (batch_size,), predictor=EulerMaruyamaManifoldPredictor, corrector=None, continuous=True, forward=False, eps=1e-3)
    # p_train_state = replicate(train_state)
    # samples, _ = sampler(replicate(jax.random.PRNGKey(0)), p_train_state)
    rng, next_rng = jax.random.split(rng)
    x, _ = sampler(next_rng, train_state, t=t)
    # prior_likelihood = lambda x: jnp.exp(sde.prior_logp(x))
    score, _ = score_model.apply(train_state.params, train_state.model_state, next_rng, x=x, t=t)
    prob = jax.vmap(sde.marginal_log_prob)(x0, x, jnp.ones(x.shape[0]) * t)
    plot_and_save3(x0, x, jnp.exp(prob), score, out="images/s2_xt_backw_score.jpg")
    plot_and_save([x0, x], out=f"s2_train_{steps}.jpg")

    # score, _ = score_model.apply(train_state.params, train_state.model_state, None, x=jnp.array([[0.,1.,0.],[0.,0.,1.]]), t=t)
    # print(jnp.linalg.norm(score))
    save('experiments', 'params', train_state.params)
    save('experiments', 'state', train_state.model_state)
    # params = restore('experiments', 'params')
    # state = restore('experiments', 'state')
    # score, _ = score_model.apply(params, state, None, x=x, t=t)


if __name__ == "__main__":
    # test_score()
    # main()
    forward_or_backward_process()