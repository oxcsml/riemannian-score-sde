# %%
%load_ext autoreload
%autoreload 2

import functools

import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk
import optax

from score_sde.models import MLP
from score_sde.utils import TrainState, get_data_inverse_scaler
from score_sde.sde import subVPSDE
from score_sde.losses import get_step_fn
# %%
score_model = hk.transform_with_state(
    MLP(hidden_shapes=[128,128,128], output_shape=1, act='swish')
)
dummy_input = jnp.zeros((10,10))
params, state = score_model.init(rng=jax.random.PRNGKey(0), x=dummy_input)
# %%
# warmup = optax.linear_schedule(init_value=0.0, end_value=)

optimiser = optax.chain(
    optax.clip_by_global_norm(jnp.inf),
    optax.adam(1e-3)
)

opt_state = optimiser.init(params)
# %%
train_state = TrainState(
    opt_state=opt_state, model_state=state, step=0, params=params, ema_rate=0.999, params_ema=params, rng=jax.random.PRNGKey(0)
)
# %%
sde = subVPSDE(
    beta_min=0.1,
    beta_max=20.,
    N=1000
)
# %%
train_step_fn = get_step_fn(sde, score_model.apply, True, reduce_mean=False, continuous=True, likelihood_weighting=False)
p_train_step = jax.jit(functools.partial(jax.lax.scan, train_step_fn))
# %%
batch = {
    'data': jnp.array(np.random.uniform(size=(10,3,32,32)))
}

p_train_step((jax.random.PRNGKey(0),train_state), batch)
# %%
