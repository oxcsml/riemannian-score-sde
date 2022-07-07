# %%
%load_ext autoreload
%autoreload 2

import os
os.environ["GEOMSTATS_BACKEND"] = 'jax'
os.chdir('/data/localhost/not-backed-up/mhutchin/score-sde')

import jax
# %%

from riemannian_score_sde.datasets import *
from riemannian_score_sde.utils.vis import setup_sphere_plot, scatter_earth
from score_sde.datasets import DataLoader, SubDataset, TensorDataset, random_split

# %%

data = VolcanicErruption()
fig, ax = setup_sphere_plot()
scatter_earth(data.data, ax=ax)

# %%

data = Fire()
fig, ax = setup_sphere_plot(azim=-45, elev=45)
scatter_earth(data.data, ax=ax)

# %%

data = Flood()
fig, ax = setup_sphere_plot()
scatter_earth(data.data, ax=ax)

# %%

data = Earthquake()
fig, ax = setup_sphere_plot(azim=90, elev=-0)
scatter_earth(data.data, ax=ax)
# ax.set_aspect('equal')

# %%
dataloader = DataLoader(Earthquake(), 100, jax.random.PRNGKey(0))
for batch in dataloader:
    print(batch.shape)

# %%
len(SubDataset(Earthquake(), jnp.arange(100)))
# %%
td = TensorDataset(jnp.arange(100)[:, None])
subset = SubDataset(td, jnp.arange(50))

for b in DataLoader(subset, 10, jax.random.PRNGKey(0), shuffle=False):
    print(b)
# %%

print([len(ds) for ds in random_split(td, [80,10,10], jax.random.PRNGKey(0))])
# %%
