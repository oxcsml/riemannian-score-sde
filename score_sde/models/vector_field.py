import abc
from typing import Sequence

import jax
import haiku as hk
import jax.numpy as jnp

from hydra.utils import instantiate
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.base import VectorSpace, EmbeddedManifold
from .flow import div_noise


def get_div_fn(fi_fn, Xi, hutchinson_type: str):
    """Pmapped divergence of the drift function."""
    if hutchinson_type == "None":
        return lambda x, t, eps: get_exact_div_fn(fi_fn, Xi)(x, t)
    else:
        return lambda x, t, eps: get_estimate_div_fn(fi_fn, Xi)(x, t, eps)


def get_estimate_div_fn(fi_fn, Xi=None):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x: jnp.ndarray, t: float, eps: jnp.ndarray):
        # grad_fn = lambda data: jnp.sum(fi_fn(data, t) * eps)
        # grad_fn_eps = jax.grad(grad_fn)(x)
        def grad_fn(data):
            fi = fi_fn(data, t)
            return jnp.sum(fi * eps), fi

        (_, fi), grad_fn_eps = jax.value_and_grad(grad_fn, has_aux=True)(x)
        # out = grad_fn_eps * G(x) @ Xi * eps
        # G = manifold.metric.metric_matrix(x)
        # Xi = jnp.einsum('...ij,...jk->...ik', G, Xi)
        if Xi is not None:
            grad_fn_eps = jnp.einsum("...d,...dn->...n", grad_fn_eps, Xi)
        div = jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))
        return div, fi

    return div_fn


def get_exact_div_fn(fi_fn, Xi=None):
    "flatten all but the last axis and compute the true divergence"

    def div_fn(
        x: jnp.ndarray,
        t: float,
    ):
        if len(t.shape) == len(x.shape) - 1:
            # Assume t is just missing the last dim of x
            t = jnp.expand_dims(t, axis=-1)

        x_shape = x.shape
        x = jnp.expand_dims(x.reshape((-1, x_shape[-1])), 1)
        t = jnp.expand_dims(t.reshape((-1, t.shape[-1])), 1)
        jac = jax.vmap(jax.jacrev(fi_fn, argnums=0))(x, t)
        jac = jac.reshape([*x_shape[:-1], -1, x_shape[-1]])
        # G = manifold.metric.metric_matrix(x)
        # Xi = jnp.einsum('...ij,...jk->...ik', G, Xi(x))
        if Xi is not None:
            jac = jnp.einsum("...nd,...dm->...nm", jac, Xi)
        div = jnp.trace(jac, axis1=-1, axis2=-2).reshape(x_shape[:-1])
        return div

    return div_fn


class VectorFieldGenerator(hk.Module, abc.ABC):
    def __init__(self, architecture, embedding, output_shape, manifold):
        """X = fi * Xi with fi weights and Xi generators"""
        super().__init__()
        self.net = instantiate(architecture, output_shape=output_shape)
        self.embedding = instantiate(embedding, manifold=manifold)
        self.manifold = manifold

    @staticmethod
    @abc.abstractmethod
    def output_shape(manifold):
        """Cardinality of the generating set."""

    def _weights(self, x, t):
        """shape=[..., card=n]"""
        return self.net(*self.embedding(x, t))

    @abc.abstractmethod
    def _generators(self, x):
        """Set of generating vector fields: shape=[..., d, card=n]"""

    @property
    def decomposition(self):
        return lambda x, t: self._weights(x, t), lambda x: self._generators(x)

    def __call__(self, x, t):
        fi_fn, Xi_fn = self.decomposition
        fi, Xi = fi_fn(x, t), Xi_fn(x)
        out = jnp.einsum("...n,...dn->...d", fi, Xi)
        # seems that extra projection is required for generator=eigen
        # during the ODE solve cf tests/test_lkelihood.py
        out = self.manifold.to_tangent(out, x)
        return out

    def div_generators(self, x):
        """Divergence of the generating vector fields: shape=[..., card=n]"""

    def div_split(self, x, t, hutchinson_type):
        """Returns div(X) = Xi(fi) + fi div(Xi)"""
        fi_fn, Xi_fn = self.decomposition
        Xi = Xi_fn(x)
        if hutchinson_type == "None":
            # splitting div is unecessary when computated exactly
            # term_1 = get_exact_div_fn(fi_fn, Xi)(x, t)
            out = get_exact_div_fn(self.__call__, None)(x, t)
        else:
            shape = [*x.shape[:-1], self.output_shape(self.manifold)]
            eps = div_noise(hk.next_rng_key(), shape, hutchinson_type)
            term_1, fi = get_estimate_div_fn(fi_fn, Xi)(x, t, eps)
            div_Xi = self.div_generators(x)
            term_2 = jnp.einsum("...n,...n->...", fi, div_Xi)
            out = term_1 + term_2
        return out

    def divE(self, x, t, hutchinson_type):
        """Euclidean divergence cf Rozen et al. 2021"""
        if hutchinson_type == "None":
            out = get_exact_div_fn(self.__call__, None)(x, t)
        else:
            shape = [*x.shape[:-1], self.output_shape(self.manifold)]
            eps = div_noise(hk.next_rng_key(), shape, hutchinson_type)
            out, _ = get_estimate_div_fn(self.__call__, None)(x, t, eps)
        return out


class DivFreeGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)
        self.div = self.div_split

    @staticmethod
    def output_shape(manifold):
        return manifold.isom_group.dim

    def _generators(self, x):
        return self.manifold.div_free_generators(x)

    def div_generators(self, x):
        shape = [*x.shape[:-1], self.output_shape(self.manifold)]
        return jnp.zeros(shape)


class EigenGenerator(VectorFieldGenerator):
    """Gradient of laplacien eigenfunctions with eigenvalue=1"""

    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)
        assert isinstance(manifold, Hypersphere)
        self.div = self.div_split

    @staticmethod
    def output_shape(manifold):
        return manifold.embedding_space.dim

    def _generators(self, x):
        return self.manifold.eigen_generators(x)

    def div_generators(self, x):
        # NOTE: Empirically need this factor 2 to match AmbientGenerator but why??
        return -self.manifold.dim * 2 * x


class AmbientGenerator(VectorFieldGenerator):
    """Equivalent to EigenGenerator"""

    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)
        self.div = self.divE

    @staticmethod
    def output_shape(manifold):
        if isinstance(manifold, EmbeddedManifold):
            output_shape = manifold.embedding_space.dim
        else:
            output_shape = manifold.dim
        return output_shape

    def _generators(self, x):
        return self.manifold.eigen_generators(x)

    def __call__(self, x, t):
        # `to_tangent`` have an 1/sq_norm(x) term that wrongs the div
        return self.manifold.to_tangent(self.net(x, t), x)
