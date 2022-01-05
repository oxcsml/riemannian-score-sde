from functools import partial

from geomstats.geometry.hypersphere import Hypersphere
import geomstats.backend as gs

from score_sde.sampling import EulerMaruyamaManifoldPredictor, get_pc_sampler
from score_sde.sde import SDE, batch_mul, Brownian
from score_sde.utils import TrainState, ScoreFunctionWrapper, replicate
from score_sde.models import MLP
from score_sde.losses import get_pmap_step_fn, get_step_fn
import jax
from jax import numpy as jnp
import haiku as hk
import optax
import numpy as np

from scripts.utils import plot_and_save, plot_and_save_video, vMF_pdf


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
    

@partial(jax.jit, static_argnums=(-1))
def brownian_motion(previous_x, traj, delta, N, manifold):
    rng = jax.random.PRNGKey(0)

    def body(step, val):
        rng, x, traj = val
        traj = traj.at[step].set(x)
        rng, tangent_noise = manifold.random_normal_tangent(state=rng, base_point=x, n_samples=x.shape[0])
        x = manifold.metric.exp(tangent_vec=delta * tangent_noise, base_point=x)
        return rng, x, traj

    _, x, traj = jax.lax.fori_loop(0, N, body, (rng, previous_x, traj))
    return x, traj


def forward_process():
    S2 = Hypersphere(dim=2)
    N = 100
    n_samples = 1000
    delta = 0.1
    mu = gs.array([1, 0, 0])
    kappa = 15
    initial_point = S2.random_von_mises_fisher(kappa=kappa, n_samples=n_samples)
    previous_x = initial_point
    traj = jnp.zeros((N + 1, previous_x.shape[0], S2.dim + 1))

    x, traj = brownian_motion(previous_x, traj, delta, N, S2)
    plot_and_save_video(traj, pdf=partial(vMF_pdf, mu=mu, kappa=kappa), out="forward.mp4")




def main():
    manifold = Hypersphere(dim=2)
    N = 1000
    n_samples = 10 ** 3
    batch_size = 256

    dataset = vMFDataset([batch_size], jax.random.PRNGKey(0), manifold, kappa=15)
    x = next(dataset)
    # x = manifold.random_von_mises_fisher(kappa=15, n_samples=n_samples)

    sde = Brownian(manifold, T=5, N=N)
    # x = sde.prior_sampling(jax.random.PRNGKey(0), (n_samples,))
    # timesteps = jnp.linspace(sde.T, 1e-3, sde.N)
    # predictor = EulerMaruyamaManifoldPredictor(sde, score_fn=None)

    # Score networks
    # score_model = lambda x, t: gs.concatenate([-x[..., [1]], x[..., [0]], gs.zeros((*x.shape[:-1], 1))], axis=-1)  # one of the divergence free vector field, i.e. generate an isometry
    # score_model = lambda x, t: manifold.to_tangent(gs.concatenate([gs.zeros((*x.shape[:-1], 1)), 10 * gs.ones((*x.shape[:-1], 1)), gs.zeros((*x.shape[:-1], 1))], axis=-1), x)
    def score_model(x, t):
        # t = gs.expand_dims(t, axis=(-1, -2))
        # print(t.shape)
        return manifold.to_tangent(ScoreFunctionWrapper(MLP(hidden_shapes=3*[128], output_shape=3, act='sin'))(x, t), x)  # NOTE: regularize orthogonal component?
    # def score_model(x, t):
    #     invariant_basis = manifold.invariant_basis(x)
    #     weights = ScoreFunctionWrapper(MLP(hidden_shapes=1*[64], output_shape=3, act='sin'))(x, t)  # output_shape = dim(Isom(manifold))
    #     return (gs.expand_dims(weights, -2) * invariant_basis).sum(-1)

    score_model = hk.transform_with_state(score_model)
    params, state = score_model.init(rng=jax.random.PRNGKey(0), x=x, t=0)
    # print(score_model.apply(params, state, jax.random.PRNGKey(0), x=x, t=0)[0].shape)

    # steps = 100000 // 100
    # warmup_steps = 2000 // 100
    steps = 100000 // 1000
    warmup_steps = 2000 // 1000

    schedule_fn = optax.join_schedules([
        optax.linear_schedule(init_value=0.0, end_value=1.0, transition_steps=warmup_steps),
        optax.cosine_decay_schedule(init_value=1.0, decay_steps = steps - warmup_steps, alpha=0.0),
    ], [warmup_steps])

    lr=2e-4
    grad_clip=jnp.inf

    optimiser = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(lr, b1=.9, b2=0.999,eps=1e-8),
        optax.scale_by_schedule(schedule_fn)
    )
    opt_state = optimiser.init(params)

    train_state = TrainState(
        opt_state=opt_state, model_state=state, step=0, params=params, ema_rate=0.999, params_ema=params, rng=jax.random.PRNGKey(0)
    )
    # train_step_fn = get_pmap_step_fn(sde, score_model, optimiser, True, reduce_mean=False, continuous=True, likelihood_weighting=False)
    train_step_fn = get_step_fn(sde, score_model, optimiser, True, reduce_mean=False, continuous=True, likelihood_weighting=False)
    p_train_step = jax.pmap(partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
    p_train_state = replicate(train_state)


    # dataset = vMFDataset([1,1,batch_size], jax.random.PRNGKey(0), manifold, kappa=15)
    dataset = vMFDataset([batch_size], jax.random.PRNGKey(0), manifold, kappa=15)

    for i in range(steps):
        batch = {
            'data': next(dataset),
            'label': None
        }
        # rng = p_train_state.rng[0]
        rng = train_state.rng

        # next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        # rng = next_rng[0]
        # next_rng = next_rng[1:]
        rng, next_rng = jax.random.split(rng)
        # (_, p_train_state), loss = p_train_step((next_rng, p_train_state), batch)
        (_, train_state), loss = train_step_fn((next_rng, train_state), batch)
        # if i%100 == 0:
        print(i, ': ', loss)


    sampler = get_pc_sampler(sde, score_model, (n_samples,), predictor=EulerMaruyamaManifoldPredictor, corrector=None, inverse_scaler=lambda x: x, snr=0.2, continuous=True)
    # p_train_state = replicate(train_state)
    # samples, _ = sampler(replicate(jax.random.PRNGKey(0)), p_train_state)
    samples, _ = sampler(jax.random.PRNGKey(0), train_state)
    # prior_likelihood = lambda x: jnp.exp(sde.prior_logp(x))
    x = next(vMFDataset([n_samples], jax.random.PRNGKey(0), manifold, kappa=15))
    # traj = jnp.concatenate([x, samples], axis=0)
    traj = [x, samples]
    plot_and_save(traj, out=f"sphere_{steps}.jpg")

    # K = 100
    # rng = jax.random.PRNGKey(0)
    # rng, step_rng = jax.random.split(rng)
    # x0 = sde.prior_sampling(step_rng, (K,))
    # rng, step_rng = jax.random.split(rng)
    # print("x0", x0.shape)
    # rng, step_rng = jax.random.split(rng)
    # # t = jax.random.uniform(step_rng, (K,), minval=1e-3, maxval=sde.T)
    # t = 1.
    # print("t", t)
    # rng, step_rng = jax.random.split(rng)
    # x = sde.marginal_sample(step_rng, x0, t, train_state)
    # # print(manifold.metric.dist_broadcast(x0, x).mean())
    # # x = sde.marginal_sample(replicate(step_rng), replicate(x0), replicate(t), p_train_state)
    # print("x", x.shape)
    # # x = x.squeeze(0)
    # # x = sde.marginal_sample(step_rng, x0, 5, train_state)


if __name__ == "__main__":
    main()