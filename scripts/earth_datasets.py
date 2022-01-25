# %%
%load_ext autoreload
%autoreload 2

import os
os.environ["GEOMSTATS_BACKEND"] = 'jax'
os.chdir('/data/localhost/not-backed-up/mhutchin/score-sde')

import jax
# %%

from riemannian_score_sde.datasets import *
from score_sde.utils.vis import setup_sphere_plot, scatter_earth


# %%

data = VolcanicErruption((100,), jax.random.PRNGKey(0))
fig, ax = setup_sphere_plot()
scatter_earth(data.data, ax=ax)

# %%

data = Fire((100,), jax.random.PRNGKey(0))
fig, ax = setup_sphere_plot(azim=-45, elev=45)
scatter_earth(data.data, ax=ax)

# %%

data = Flood((100,), jax.random.PRNGKey(0))
fig, ax = setup_sphere_plot()
scatter_earth(data.data, ax=ax)

# %%

data = Earthquake((100,), jax.random.PRNGKey(0))
fig, ax = setup_sphere_plot(azim=90, elev=-0)
scatter_earth(data.data, ax=ax)
# ax.set_aspect('equal')

# %%
import matplotlib.pyplot as plt
plt.scatter(data.intrinsic_data[..., 1], data.intrinsic_data[..., 0],s=0.5)
plt.gca().set_aspect('equal')


# %%
data = Earthquake((1000,), jax.random.PRNGKey(0))
for batch in data:
    print(batch.shape)

# %%
