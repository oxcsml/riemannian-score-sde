# %%
%load_ext autoreload
%autoreload 2

import functools

import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk
import optax

import matplotlib.pyplot as plt

from score_sde.datasets import GaussianMixture
from score_sde.models import MLP
from score_sde.sampling import get_pc_sampler, get_predictor, get_corrector, get_ode_sampler
from score_sde.likelihood import get_likelihood_fn
from score_sde.utils import TrainState, get_data_inverse_scaler, ScoreFunctionWrapper, replicate
from score_sde.sde import subVPSDE, VPSDE
from score_sde.losses import get_pmap_step_fn, get_step_fn
# %%
batch_size = 256
# dataset = GaussianMixture([batch_size], jax.random.PRNGKey(0), stds=[0.1,0.1])
a = 5
dataset = GaussianMixture([batch_size], jax.random.PRNGKey(0), means=[a], stds=[1.0], weights=[1.])

def score_1d_gaussian(x, t):
    t = jnp.array(t)
    if len(t.shape) == 0:
        t = t * jnp.ones(x.shape[:-1])

    if len(t.shape) == len(x.shape) - 1:
        t = jnp.expand_dims(t, axis=-1)

    return jnp.ones_like(x) * a * jnp.exp(-2 * t)

score_model = hk.transform_with_state(score_1d_gaussian)

# score_model = hk.transform_with_state(
#     lambda x, t: jnp.zeros_like(x) - 0.5
#     lambda x, t: ScoreFunctionWrapper(MLP(hidden_shapes=3*[128], output_shape=1, act='sin'))(x, t)
# )

dummy_input = next(dataset)
params, state = score_model.init(rng=jax.random.PRNGKey(0), x=dummy_input, t=0)

out = score_model.apply(params, state, jax.random.PRNGKey(0), x=dummy_input, t=0)
# %%

# steps = 100000
# warmup_steps = 2000
steps = 100000 // 100
warmup_steps = 2000 // 100

schedule_fn = optax.join_schedules([
    optax.linear_schedule(init_value=0.0, end_value=1.0, transition_steps=warmup_steps),
    # optax.linear_schedule(init_value=1.0, end_value=1.0, transition_steps=steps - warmup_steps),
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

# %%
# lrs = jnp.array([
#     schedule_fn(step) for step in range(steps)
# ])
# plt.plot(lrs)

# %%
train_state = TrainState(
    opt_state=opt_state, model_state=state, step=0, params=params, ema_rate=0.999, params_ema=params, rng=jax.random.PRNGKey(0)
)
p_train_state = replicate(train_state)
# %%
sde = VPSDE(
    beta_min=0.1,
    beta_max=20.,
    N=1000
)
# %%
train_step_fn = get_pmap_step_fn(sde, score_model, optimiser, True, reduce_mean=False, continuous=True, like_w=False)
p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
# %%

batch_size = 256
# dataset = GaussianMixture([1,1,batch_size], jax.random.PRNGKey(0), stds=[0.1,0.1])
dataset = GaussianMixture([1,1,batch_size], jax.random.PRNGKey(0), means=[a], stds=[1.0], weights=[1.])

# # %%
# losses = []
# for i in range(steps):
#     batch = {
#         'data': next(dataset),
#         'label': None
#     }
#     rng = p_train_state.rng[0]
#     next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
#     rng = next_rng[0]
#     next_rng = next_rng[1:]
#     (_, p_train_state), loss = p_train_step((next_rng,p_train_state), batch)
#     losses.append(float(loss))
#     if i%100 == 0:
#         print(i, ': ', loss)

# print(loss)
# losses = jnp.array(losses)

# plt.plot(losses[::10])
# plt.yscale('log')
# %%
sampler = get_pc_sampler(sde, score_model, (2**12,1), get_predictor("EulerMaruyamaPredictor"), get_corrector("NoneCorrector"), lambda x: x, 0.2)
# sampler = get_ode_sampler(sde, score_model, (2**12,1), lambda x: x)
posterior_samples = sampler(replicate(jax.random.PRNGKey(0)), p_train_state)

# %%
target_samples = next(dataset)
import seaborn as sns
# sns.kdeplot(posterior_samples[0][0,:,0], color='tab:orange')
# sns.kdeplot(target_samples[0,0,:,0], color='tab:green')

likelihood_fn = get_likelihood_fn(sde, score_model, lambda x: x, bits_per_dimension=False)

x = jnp.linspace(-5, 5)[np.newaxis, :, np.newaxis]

prior_likelihood = jnp.exp(sde.prior_logp(x[0]))
pushforward_likelihood = jnp.exp(likelihood_fn(replicate(jax.random.PRNGKey(0)), p_train_state, x)[0])

plt.plot(x[0,:,0], prior_likelihood, color='tab:blue')
sns.kdeplot(target_samples[0,0,:,0], color='tab:green')
plt.plot(x[0,:,0], pushforward_likelihood[0], color='tab:orange')
sns.kdeplot(posterior_samples[0][0,:,0], color='tab:red')

# %%
