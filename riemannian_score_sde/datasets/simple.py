import math
import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf as normal_pdf
import numpy as np
from scipy.special import ive

import geomstats.backend as gs
from geomstats.geometry.lie_group import LieGroup
from geomstats.geometry.special_orthogonal import _SpecialOrthogonal3Vectors
from geomstats.algebra_utils import from_vector_to_diagonal_matrix

from riemannian_score_sde.models.distribution import (
    WrapNormDistribution as WrappedNormal,
)


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


class vMFDataset:
    def __init__(self, batch_dims, rng, manifold, mu, kappa, **kwargs):
        self.manifold = manifold
        self.d = self.manifold.dim + 1
        self.mu = jnp.array(mu)
        assert manifold.belongs(self.mu)
        self.kappa = jnp.array([kappa])
        self.batch_dims = batch_dims
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.manifold.random_von_mises_fisher(
            mu=self.mu, kappa=self.kappa, n_samples=np.prod(self.batch_dims)
        )
        batch_dims = (self.batch_dims,) if isinstance(self.batch_dims, int) else self.batch_dims
        samples = samples.reshape([*batch_dims, samples.shape[-1]])

        return (samples, None)

    def _log_normalization(self):
        output = -(
            (self.d / 2 - 1) * jnp.log(self.kappa)
            - (self.d / 2) * math.log(2 * math.pi)
            - (self.kappa + jnp.log(ive(self.d / 2 - 1, self.kappa)))
        )
        return output.reshape([1, *output.shape[:-1]])

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.kappa * (jnp.expand_dims(self.mu, 0) * x).sum(-1, keepdims=True)
        return output.reshape([*output.shape[:-1]])

    def entropy(self):
        output = (
            -self.kappa
            * ive(self.d / 2, self.kappa)
            / ive((self.d / 2) - 1, self.kappa)
        )
        return output.reshape([*output.shape[:-1]]) + self._log_normalization()


class DiracDataset:
    def __init__(self, batch_dims, mu, **kwargs):
        self.mu = jnp.array(mu)
        self.batch_dims = batch_dims

    def __iter__(self):
        return self

    def __next__(self):
        n_samples = np.prod(self.batch_dims)
        samples = jnp.repeat(self.mu.reshape(1, -1), n_samples, 0)
        return (samples, None)


class WrapNormDistribution:
    def __init__(self, batch_dims, manifold, scale=1.0, mean=None, seed=0, rng=None):
        self.manifold = manifold
        self.batch_dims = batch_dims
        if mean is None:
            mean = jnp.zeros(manifold.dim)
        mean = jnp.array(mean)
        self.dist = WrappedNormal(manifold, scale, mean)
        self.rng = rng if rng is not None else jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        self.rng, rng = jax.random.split(self.rng)
        return self.dist.sample(rng, self.batch_dims), None


class Wrapped:
    def __init__(
        self,
        scale,
        scale_type,
        K,
        batch_dims,
        manifold,
        seed,
        conditional,
        mean,
        **kwargs,
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
        elif mean == "anti":
            v = jnp.array([[jnp.pi, 0.0, 0.0]])
            self.mean = _SpecialOrthogonal3Vectors().matrix_from_tait_bryan_angles(v)
        elif mean == "id" and isinstance(self.manifold, LieGroup):
            self.mean = self.manifold.identity
        else:
            raise ValueError(f"Mean value: {mean}")

        if scale_type == "random":
            precision = jax.random.gamma(key=next_rng, a=scale, shape=(K,))
        elif scale_type == "fixed":
            precision = jnp.ones((K,)) * (1 / scale**2)
        else:
            raise ValueError(f"Scale value: {scale}")
        axis_to_expand = tuple(range(-1, -len(self.mean.shape), -1))
        self.precision = jnp.expand_dims(precision, axis_to_expand)

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

    def log_prob(self, samples):
        # TODO: this is wrong, cf WrapNormDistribution in distribution.py
        def single_log_prob(samples, mean, precision):
            pos = self.manifold.log(samples, mean)
            ll = normal_pdf(pos, scale=1 / jnp.sqrt(precision))
            return ll.prod(axis=-1)

        ll = jax.vmap(single_log_prob, (None, 0, 0), (0))(
            samples, self.mean, self.precision
        )
        return ll.mean(axis=0)


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
            i += 1

        if self.conditional:
            return samples, jnp.expand_dims(k, -1)
        else:
            return (samples, None)
