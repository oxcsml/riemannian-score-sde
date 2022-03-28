import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.algebra_utils import from_vector_to_diagonal_matrix 
import jax
import jax.numpy as jnp
import numpy as np


class Uniform:
    def __init__(
        self, batch_dims, manifold, seed, **kwargs
    ):
        self.batch_dims = batch_dims
        self.manifold = manifold
        self.rng = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        rng, next_rng = jax.random.split(self.rng)
        self.rng = rng
        n_samples=np.prod(self.batch_dims)
        samples = self.manifold.random_uniform(state=next_rng, n_samples=n_samples)
        return (samples, None)


class Wrapped:
    def __init__(
        self, scale, K, batch_dims, manifold, seed, conditional, **kwargs
    ):
        self.K = K
        self.scale = scale
        self.batch_dims = batch_dims
        self.manifold = manifold
        rng = jax.random.PRNGKey(seed)
        rng, next_rng = jax.random.split(rng)
        self.rng = rng
        self.conditional = conditional
        if 'mean' in kwargs:
            self.mean = kwargs["mean"]
        else:
            self.mean = self.manifold.random_uniform(state=next_rng, n_samples=K)
            # self.mean = self.manifold.identity

    def __iter__(self):
        return self

    def __next__(self):
        rng, next_rng = jax.random.split(self.rng)
        self.rng = rng
        n_samples=np.prod(self.batch_dims)
        # _, mean = gs.random.choice(state=next_rng, a=self.mean, n=n_samples)
        ks = jnp.arange(self.mean.shape[0])
        _, k = gs.random.choice(state=next_rng, a=ks, n=n_samples)
        mean = self.mean[k]
        tangent_vec = self.manifold.random_normal_tangent(
            state=next_rng, base_point=mean, n_samples=n_samples
        )[1]
        tangent_vec = self.scale * tangent_vec
        samples = self.manifold.exp(tangent_vec, mean)
        if self.conditional:
            return samples, jnp.expand_dims(k, -1)
        else:
            return (samples, None)


class Langevin:
    """
    https://dr.lib.iastate.edu/server/api/core/bitstreams/66c3ca3b-75b3-4946-bbb1-3576c0334489/content
    https://arxiv.org/pdf/0712.4166.pdf
    """
    def __init__(
        self, beta, k, batch_dims, manifold, seed, **kwargs
    ):
        self.beta = beta
        self.batch_dims = batch_dims
        self.manifold = manifold
        rng = jax.random.PRNGKey(seed)
        rng, next_rng = jax.random.split(rng)
        self.rng = rng
        if 'mean' in kwargs:
            self.mean = kwargs["mean"]
        else:
            self.mean = self.manifold.random_uniform(state=next_rng, n_samples=k)

    def __iter__(self):
        return self

    def __next__(self):
        rng, next_rng = jax.random.split(self.rng)
        self.rng = rng
        n_samples=np.prod(self.batch_dims)

        _, C = gs.random.choice(state=next_rng, a=self.mean, n=n_samples)
        C_tr = gs.transpose(C, axes=(0, 2, 1))
        _, D, _ = jnp.linalg.svd(C)
        D = from_vector_to_diagonal_matrix(D)

        cond = jnp.zeros(n_samples)
        samples = jnp.zeros(n_samples, self.manifold.n, self.manifold.n)

        while not cond.all():
            X = self.manifold.random_uniform(state=next_rng, n_samples=n_samples)
            thresh = gs.exp(gs.trace(C_tr @ X - D, axis1=1, axis2=2))
            rng, next_rng = jax.random.split(rng)
            _, u = gs.random.rand(state=next_rng, size=n_samples)
            mask = u < thresh
            mask = gs.expand_dims(mask, axis=(-1, -2))
            samples = (1 - mask) * samples + mask * X
            cond = (1 - mask) * cond + mask * mask
            
        return (samples, None)