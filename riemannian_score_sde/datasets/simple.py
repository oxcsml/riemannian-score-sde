import geomstats.backend as gs
from geomstats.geometry.lie_group import LieGroup
from geomstats.geometry.special_orthogonal import (
    SpecialOrthogonal,
    _SpecialOrthogonal3Vectors,
)
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
import jax
import jax.numpy as jnp
import numpy as np


class Uniform:
    def __init__(self, batch_dims, manifold, seed, **kwargs):
        self.batch_dims = batch_dims
        self.manifold = manifold
        self.rng = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        rng, next_rng = jax.random.split(self.rng)
        self.rng = rng
        n_samples = np.prod(self.batch_dims)
        samples = self.manifold.random_uniform(state=next_rng, n_samples=n_samples)
        return (samples, None)


class Wrapped:
    def __init__(
        self, scale, K, batch_dims, manifold, seed, conditional, mean, **kwargs
    ):
        self.K = K
        self.batch_dims = batch_dims
        self.manifold = manifold
        rng = jax.random.PRNGKey(seed)
        rng, next_rng = jax.random.split(rng)
        self.rng = rng
        self.conditional = conditional
        if mean == "unif":
            self.mean = self.manifold.random_uniform(state=next_rng, n_samples=K)
        elif mean == "anti" and isinstance(self.manifold, _SpecialOrthogonal3Vectors):
            v = jnp.array([[jnp.pi, 0.0, 0.0]])
            self.mean = _SpecialOrthogonal3Vectors().matrix_from_tait_bryan_angles(v)
        elif mean == "id" and isinstance(self.manifold, LieGroup):
            self.mean = self.manifold.identity
        else:
            raise ValueError(f"Mean value: {mean}")

        if scale == "random":
            precision = jax.random.gamma(key=next_rng, a=100.0, shape=(K,))
        elif scale == "fixed":
            precision = jnp.ones((K,)) * (1 / 0.2**2)
        else:
            raise ValueError(f"Scale value: {scale}")
        self.precision = jnp.expand_dims(precision, (-1, -2))

    def __iter__(self):
        return self

    def __next__(self):
        n_samples = np.prod(self.batch_dims)
        ks = jnp.arange(self.mean.shape[0])
        rng, next_rng = jax.random.split(self.rng)
        self.rng = rng
        _, k = gs.random.choice(state=next_rng, a=ks, n=n_samples)
        mean = self.mean[k]
        scale = 1 / jnp.sqrt(self.precision[k])
        tangent_vec = self.manifold.random_normal_tangent(
            state=next_rng, base_point=mean, n_samples=n_samples
        )[1]
        tangent_vec = scale * tangent_vec
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

    def __init__(self, scale, K, batch_dims, manifold, seed, conditional, **kwargs):
        self.batch_dims = batch_dims
        self.manifold = manifold
        rng = jax.random.PRNGKey(seed)
        rng, next_rng = jax.random.split(rng)
        self.rng = rng
        self.conditional = conditional
        if "mean" in kwargs:
            self.mean = kwargs["mean"]
        else:
            self.mean = self.manifold.random_uniform(state=next_rng, n_samples=K)
        self.precision = jax.random.gamma(key=next_rng, a=scale, shape=(K,))

    def __iter__(self):
        return self

    def __next__(self):
        n_samples = np.prod(self.batch_dims)
        ks = jnp.arange(self.mean.shape[0])
        rng, next_rng = jax.random.split(self.rng)
        self.rng = rng
        _, k = gs.random.choice(state=next_rng, a=ks, n=n_samples)
        C = self.mean[k]
        kappa = self.precision[k]
        C_tr = gs.transpose(C, axes=(0, 2, 1))
        _, D, _ = jnp.linalg.svd(C)
        D = from_vector_to_diagonal_matrix(D)

        cond = jnp.zeros(n_samples)
        samples = jnp.zeros((n_samples, self.manifold.n, self.manifold.n))
        i = 0
        while not cond.all():
            X = self.manifold.random_uniform(state=next_rng, n_samples=n_samples)
            thresh = gs.exp(kappa * gs.trace(C_tr @ X - D, axis1=1, axis2=2))
            rng, next_rng = jax.random.split(rng)
            _, u = gs.random.rand(state=next_rng, size=n_samples)
            mask = u < thresh
            mask = gs.expand_dims(mask, axis=(-1, -2))
            samples = (1 - mask) * samples + mask * X
            cond = (1 - mask) * cond + mask * mask
            # print("iter", i)
            # print("mask", mask.mean().item())
            # print("cond", cond.mean().item())
            i += 1
        print("iter", i)

        if self.conditional:
            return samples, jnp.expand_dims(k, -1)
        else:
            return (samples, None)
