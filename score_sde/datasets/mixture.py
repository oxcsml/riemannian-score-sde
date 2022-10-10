from functools import partial
import jax
import jax.numpy as jnp
import numpy as np


class GaussianMixture:
    def __init__(
        self,
        batch_dims,
        rng,
        means=[-1.0, 1.0],
        stds=[1.0, 1.0],
        weights=[0.5, 0.5],
        **kwargs
    ):
        self.means = jnp.array(means)
        self.stds = jnp.array(stds)
        self.weights = jnp.array(weights)
        self.batch_dims = batch_dims
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        rng = jax.random.split(self.rng, num=3)

        self.rng = rng[0]
        choice_key = rng[1]
        normal_key = rng[2]

        indices = jax.random.choice(
            choice_key, a=len(self.weights), shape=self.batch_dims, p=self.weights
        )
        samples = self.means[indices] + self.stds[indices] * jax.random.normal(
            normal_key, shape=self.batch_dims + self.means.shape[1:]
        )

        return (samples, None)
