# %%
%load_ext autoreload
%autoreload 2

import jax
import jax.numpy as jnp
import numpy as np

from score_sde.utils import get_div_fn

# %%


def f(x, t):
    return x ** 2 + 3 * x + 4 + t ** 2 + 3 * t


# %%

div_f = jax.jit(get_div_fn(f))
# %%
x = jnp.linspace(-1, 1)[np.newaxis, :, np.newaxis]
t = 2 * jnp.ones((50,))[np.newaxis, :]
# %%
div_f(x, t).shape
# %%
