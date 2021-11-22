# %%
%load_ext autoreload
%autoreload 2

# %%
import jax
import jax.numpy as jnp
import haiku as hk
from score_sde.models import DDPM

# %%

model = hk.transform(
    DDPM(
        sigma_min=0.1,
        sigma_max=90.0,
        num_scales=1000,
        channel_multiplier=(1, 2, 2, 2),
        num_res_blocks=2,
        centred_data=True,
        conditional=False,
    )
)
input_shape = (
    10,
    32,
    32,
    1,
)
label_shape = input_shape[:1]
fake_input = jnp.zeros(input_shape)
fake_label = jnp.zeros(label_shape, dtype=jnp.int32)


params = model.init(jax.random.PRNGKey(42), fake_input, fake_label)


# %%
