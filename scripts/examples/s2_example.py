from functools import partial

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.integrate._ivp.radau import E
import seaborn as sns
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
import geomstats.backend as gs

from riemannian_score_sde.sampling import get_pc_sampler
from score_sde.sde import SDE, batch_mul
from riemannian_score_sde.sde import Brownian
from score_sde.utils import TrainState, replicate
from score_sde.models import MLP, ScoreFunctionWrapper, ScoreNetwork, get_score_fn
from score_sde.losses import get_pmap_step_fn, get_step_fn
from score_sde.likelihood import get_likelihood_fn, get_pmap_likelihood_fn
import jax
from jax import device_put
from jax import numpy as jnp
import haiku as hk
import optax
import numpy as np

from scripts.utils import plot_and_save3, plot_and_save_video3, vMF_pdf, log_prob_vmf
from scripts.utils import save, restore


class vMFDataset:
    def __init__(self, batch_dims, rng, manifold, mu=None, kappa=1.0):
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
            mu=self.mu, kappa=self.kappa, n_samples=np.prod(self.batch_dims)
        )
        samples = samples.reshape([*self.batch_dims, samples.shape[-1]])

        return samples
        # return jnp.expand_dims(samples, axis=-1)


class DiracDataset:
    def __init__(self, batch_dims, mu=None):
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
    dataset = DiracDataset([batch_size], mu=jnp.array([1.0, 0.0, 0.0]))
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

    plot_and_save3(
        data, perturbed_data, jnp.exp(logp), logp_grad, out="images/s2_true_score.jpg"
    )


@partial(jax.jit, static_argnums=(4, 5, 6))
def EulerMaruyama(previous_x, traj, dt, timesteps, f_sde, sde, forward=True):
    rng = jax.random.PRNGKey(0)
    x0 = jnp.repeat(gs.array([[1.0, 0.0, 0.0]]), previous_x.shape[0], 0)
    N = len(timesteps)

    @jax.jit
    def body(step, val):
        rng, x, traj = val
        t = timesteps[step]
        t = jnp.broadcast_to(t, (x.shape[0], 1))
        sign = 1 if forward else -1
        rng, z = f_sde.manifold.random_normal_tangent(
            state=rng, base_point=x, n_samples=x.shape[0]
        )
        drift, diffusion = sde.sde(x, t)
        logp, logp_grad = f_sde.grad_marginal_log_prob(x0, x, t)
        # logp, logp_grad = value_and_grad_log_marginal_prob(x0, x, t, f_sde)
        tangent_vector = sign * drift * dt + batch_mul(diffusion, jnp.sqrt(dt) * z)
        x = f_sde.manifold.metric.exp(tangent_vec=tangent_vector, base_point=x)
        vector = diffusion**2 * logp_grad if forward else sign * drift

        traj = traj.at[step, :, :3].set(x)
        traj = traj.at[step, :, 3:6].set(vector)
        traj = traj.at[step, :, -1].set(logp)
        return rng, x, traj

    _, _, traj = jax.lax.fori_loop(0, N, body, (rng, previous_x, traj))
    return traj


# def forward_or_backward_process(forward=True):
def forward_or_backward_process(forward=False):
    N = 1000
    n_samples = 512
    S2 = Hypersphere(dim=2)
    # sde = Brownian(S2, T=1, N=N, beta_min=0.1, beta_max=20)
    # sde = Brownian(S2, T=3, N=N, beta_min=0.1, beta_max=1)
    sde = Brownian(S2, T=3, N=N, beta_min=1, beta_max=1)
    x0 = gs.array([[1.0, 0.0, 0.0]])
    x0b = jnp.repeat(x0, n_samples, 0)
    eps = 1e-3
    dt = (sde.T - eps) / sde.N

    if forward:
        S2_EulerMaruyama = partial(EulerMaruyama, f_sde=sde, sde=sde, forward=True)
        # previous_x = S2.random_von_mises_fisher(kappa=15, n_samples=n_samples)
        previous_x = next(DiracDataset([n_samples], mu=x0))
        timesteps = jnp.linspace(eps, sde.T, sde.N)
    else:
        params = restore("experiments", "params")
        state = restore("experiments", "state")
        score_model = hk.transform_with_state(partial(score_inv, sde=sde))
        # score_model = hk.transform_with_state(partial(score_true, sde=sde, x0=x0b))
        score_fn = get_score_fn(sde, score_model, params, state)
        # score_fn = lambda x, t: value_and_grad_log_marginal_prob(x0b, x, t, sde)[1]
        rsde = sde.reverse(score_fn)
        S2_EulerMaruyama = partial(EulerMaruyama, f_sde=sde, sde=rsde, forward=False)
        previous_x = Brownian(S2, T=sde.T, N=sde.N).prior_sampling(
            jax.random.PRNGKey(0), [n_samples]
        )
        timesteps = jnp.linspace(sde.T, eps, sde.N)

    traj = jnp.zeros((N, previous_x.shape[0], (S2.dim + 1) * 2 + 1))
    traj = S2_EulerMaruyama(previous_x, traj, 1 * dt, timesteps)

    idx = (jnp.arange(N) % 5) == 0
    traj_x = traj[idx, :, :3]
    traj_logp_grad = traj[idx, :, 3:6]
    traj_logp_grad = traj_logp_grad
    traj_logp = traj[idx, :, -1]
    timesteps = list(map(lambda x: f"{x:.2f}", timesteps[idx]))

    direction = "forward" if forward else "backward"
    name = f"s2_{direction}_{sde.T}_{sde.beta_0:.1f}_{sde.beta_1:.1f}"
    plot_and_save_video3(
        traj_x,
        jnp.exp(traj_logp),
        traj_logp_grad,
        timesteps,
        size=20,
        fps=10,
        dpi=50,
        out=f"images/{name}.mp4",
    )


### Score networks ###

# score_model = lambda x, t: manifold.metric.log(x0, x)


def score_true(x, t, x0, sde):
    if isinstance(t, (int, float)):
        t = jnp.ones((x.shape[0], 1)) * t
    score = sde.grad_marginal_log_prob(x0, x, t)[1]
    std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
    score = batch_mul(score, std)
    return score


def score_ambiant(x, t, sde):
    return sde.manifold.to_tangent(
        ScoreFunctionWrapper(MLP(hidden_shapes=3 * [512], output_shape=3, act="sin"))(
            x, t
        ),
        x,
    )  # NOTE: regularize orthogonal component?


# @partial(jax.jit, static_argnums=(2))
def score_inv(x, t, sde):
    """output_shape = dim(Isom(manifold))"""
    invariant_basis = sde.manifold.invariant_basis(x)
    layer = MLP(hidden_shapes=3 * [512], output_shape=3, act="sin")
    weights = ScoreFunctionWrapper(layer)(x, t)
    # weights = ScoreNetwork(output_shape=3, encoder_layers=[512], pos_dim=16, decoder_layers=3*[512])(x, t)
    # weights = ConcatSquash(output_shape=3, layer=layer)(x, t)  # output_shape = dim(Isom(manifold))
    return (gs.expand_dims(weights, -2) * invariant_basis).sum(-1)


def main(train=False):
    rng = jax.random.PRNGKey(0)
    manifold = Hypersphere(dim=2)
    # sde = Brownian(manifold, T=1, N=100, beta_min=0.1, beta_max=10)
    # sde = Brownian(manifold, T=3, N=100, beta_min=0.1, beta_max=1)
    sde = Brownian(manifold, T=3, N=100, beta_min=1.0, beta_max=1.0)
    batch_size = 512

    mu = jnp.array([[1.0, 0.0, 0.0]])
    mub = jnp.repeat(jnp.expand_dims(mu, 0), batch_size, 0)
    kappa = 15
    dataset_init = vMFDataset(
        [batch_size], jax.random.PRNGKey(0), manifold, mu=mu.reshape(3), kappa=kappa
    )
    # dataset_init = DiracDataset([batch_size], mu=mu)
    x = next(dataset_init)

    # score_model = partial(score_true, sde=sde, x0=mub)
    # score_model = partial(score_ambiant, sde=sde)
    score_model = partial(score_inv, sde=sde)
    score_model = hk.transform_with_state(score_model)
    if train:
        rng, next_rng = jax.random.split(rng)
        params, state = score_model.init(rng=next_rng, x=x, t=0)
    else:
        params = restore("experiments", "params")
        state = restore("experiments", "state")

    ### Optimiser ###

    warmup_steps, steps = 10, 500
    # warmup_steps, steps = 0, 20
    # warmup_steps, steps = 0, 1
    # steps = 1

    # lr=2e-4
    lr = 5e-4

    schedule_fn = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0.0, end_value=1.0, transition_steps=warmup_steps
            ),
            optax.cosine_decay_schedule(
                init_value=1.0, decay_steps=steps - warmup_steps, alpha=0.0
            ),
        ],
        [warmup_steps],
    )

    optimiser = optax.chain(
        optax.clip_by_global_norm(jnp.inf),
        optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8),
        optax.scale_by_schedule(schedule_fn),
    )
    opt_state = optimiser.init(params)

    rng, next_rng = jax.random.split(rng)
    train_state = TrainState(
        opt_state=opt_state,
        model_state=state,
        step=0,
        params=params,
        ema_rate=0.999,
        params_ema=params,
        rng=next_rng,
    )

    ### Training ###
    if train:

        # train_step_fn = get_pmap_step_fn(sde, score_model, optimiser, True, reduce_mean=False, continuous=True, like_w=False, eps=1e-3)
        train_step_fn = get_step_fn(
            sde,
            score_model,
            optimiser,
            True,
            reduce_mean=False,
            continuous=True,
            like_w=False,
            eps=1e-3,
        )
        # p_train_step = jax.pmap(partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
        # p_train_state = replicate(train_state)

        # dataset = vMFDataset([1,1,batch_size], jax.random.PRNGKey(0), manifold, kappa=15)
        dataset = dataset_init
        for i in range(steps):
            batch = {"data": next(dataset), "label": None}

            # rng = p_train_state.rng[0]
            # rng = train_state.rng

            # next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
            # rng = next_rng[0]
            # next_rng = next_rng[1:]
            # (rng, p_train_state), loss = p_train_step((next_rng, p_train_state), batch)
            rng, next_rng = jax.random.split(rng)
            (rng, train_state), loss = train_step_fn((next_rng, train_state), batch)
            # if i%100 == 0:
            print(i, ": ", loss)

    ### Analysis ###

    score_fn = get_score_fn(sde, score_model, train_state.params, train_state.model_state)
    x0 = next(dataset_init)
    name = f"{sde.T}_{sde.beta_0:.1f}_{sde.beta_1:.1f}"

    ## p_T
    # rng, next_rng = jax.random.split(rng)
    # x = sde.prior_sampling(next_rng, [batch_size])
    # score = score_fn(x, jnp.ones(x.shape[0]) * sde.T)
    # prob = jax.vmap(sde.marginal_log_prob)(x0, x, jnp.ones(x.shape[0]) * sde.T)
    # plot_and_save3(None, x, jnp.exp(prob), score, out=f"images/s2_xT_{name}.jpg")

    ## p_t (forward)
    t = 0.3
    # # x = sde.marginal_sample(step_rng, x0, t)  # Equivalent to below
    sampler = get_pc_sampler(
        sde,
        None,
        (batch_size,),
        predictor="GRW",
        corrector=None,
        continuous=True,
        forward=True,
        eps=1e-3,
    )
    rng, next_rng = jax.random.split(rng)
    x, _, _ = sampler(next_rng, train_state, x=x0, t=jnp.ones(x0.shape[0]) * t)
    logp, logp_grad = sde.grad_marginal_log_prob(x0, x, jnp.ones(x.shape[0]) * t)
    # logp_grad = score_true(x0, x, jnp.ones(x.shape[0]) * t, sde)
    plot_and_save3(
        x0, x, jnp.exp(logp), logp_grad, out=f"images/s2_xt_forw_true_{name}.jpg"
    )
    score = score_fn(x, jnp.ones(x.shape[0]) * t)
    plot_and_save3(x0, x, jnp.exp(logp), score, out=f"images/s2_xt_forw_{name}.jpg")

    ## p_t (backward)
    sampler = get_pc_sampler(
        sde,
        score_model,
        (batch_size,),
        predictor="GRW",
        corrector=None,
        continuous=True,
        forward=False,
        eps=1e-3,
    )
    # p_train_state = replicate(train_state)
    # samples, _ = sampler(replicate(jax.random.PRNGKey(0)), p_train_state)
    rng, next_rng = jax.random.split(rng)
    x, _, _ = sampler(next_rng, train_state, t=t)
    # prior_likelihood = lambda x: jnp.exp(sde.prior_logp(x))
    score = score_fn(x, jnp.ones(x.shape[0]) * t)
    prob = jax.vmap(sde.marginal_log_prob)(x0, x, jnp.ones(x.shape[0]) * t)
    plot_and_save3(x0, x, jnp.exp(prob), score, out=f"images/s2_xt_backw_{name}.jpg")

    ## p_0 (backward)
    t = 1e-3
    rng, next_rng = jax.random.split(rng)
    x, _, _ = sampler(next_rng, train_state, t=t)
    # prob = jax.vmap(sde.marginal_log_prob)(x0, x, jnp.ones(x.shape[0]) * t)
    likelihood_fn = get_likelihood_fn(
        sde, score_model, hutchinson_type="None", bits_per_dimension=False, eps=1e-3
    )
    logp, z, nfe = likelihood_fn(rng, train_state, x)
    print(nfe)
    prob = jnp.exp(logp)
    plot_and_save3(None, x, prob, None, out=f"images/s2_x0_backw_{name}.jpg")
    prob = jnp.exp(log_prob_vmf(x0, mub, kappa))
    plot_and_save3(None, x0, prob, None, out=f"images/s2_x0_true_{name}.jpg")

    # # score, _ = score_model.apply(train_state.params, train_state.model_state, None, x=jnp.array([[0.,1.,0.],[0.,0.,1.]]), t=t)
    # # print(jnp.linalg.norm(score))
    save("experiments", "params", train_state.params)
    save("experiments", "state", train_state.model_state)
    # # params = restore('experiments', 'params')
    # # state = restore('experiments', 'state')
    # # score, _ = score_model.apply(params, state, None, x=x, t=t)


if __name__ == "__main__":
    # test_score()
    main()
    # forward_or_backward_process()
