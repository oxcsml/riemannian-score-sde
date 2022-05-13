import abc
from typing import Sequence

import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp

from hydra.utils import instantiate
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.base import VectorSpace, EmbeddedManifold
from geomstats.geometry.matrices import Matrices
from .flow import div_noise


def get_div_fn(fi_fn, Xi, hutchinson_type: str):
    """Pmapped divergence of the drift function."""
    if hutchinson_type == "None":
        return lambda x, t, z, eps: get_exact_div_fn(fi_fn, Xi)(x, t, z)
    else:
        return lambda x, t, z, eps: get_estimate_div_fn(fi_fn, Xi)(x, t, z, eps)


# def get_estimate_div_fn(fi_fn, Xi=None):
#     """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

#     def div_fn(x: jnp.ndarray, t: float, eps: jnp.ndarray):
#         # grad_fn = lambda data: jnp.sum(fi_fn(data, t) * eps)
#         # grad_fn_eps = jax.grad(grad_fn)(x)
#         def grad_fn(data):
#             fi = fi_fn(data, t)
#             return jnp.sum(fi * eps), fi

#         (_, fi), grad_fn_eps = jax.value_and_grad(grad_fn, has_aux=True)(x)
#         # out = grad_fn_eps * G(x) @ Xi * eps
#         # G = manifold.metric.metric_matrix(x)
#         # Xi = jnp.einsum('...ij,...jk->...ik', G, Xi)
#         if Xi is not None:
#             grad_fn_eps = jnp.einsum("...d,...dn->...n", grad_fn_eps, Xi)
#         div = jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(x.shape))))
#         return div, fi

#     return div_fn


# def get_exact_div_fn(fi_fn, Xi=None):
#     "flatten all but the last axis and compute the true divergence"

#     def div_fn(
#         x: jnp.ndarray,
#         t: float,
#     ):
#         if len(t.shape) == len(x.shape) - 1:
#             # Assume t is just missing the last dim of x
#             t = jnp.expand_dims(t, axis=-1)

#         x_shape = x.shape
#         x = jnp.expand_dims(x.reshape((-1, x_shape[-1])), 1)
#         t = jnp.expand_dims(t.reshape((-1, t.shape[-1])), 1)
#         jac = jax.vmap(jax.jacrev(fi_fn, argnums=0))(x, t)
#         jac = jac.reshape([*x_shape[:-1], -1, x_shape[-1]])
#         # G = manifold.metric.metric_matrix(x)
#         # Xi = jnp.einsum('...ij,...jk->...ik', G, Xi(x))
#         if Xi is not None:
#             jac = jnp.einsum("...nd,...dm->...nm", jac, Xi)
#         div = jnp.trace(jac, axis1=-1, axis2=-2).reshape(x_shape[:-1])
#         return div

#     return div_fn


def get_exact_div_fn(fi_fn, Xi=None):
    "flatten all but the last axis and compute the true divergence"

    def div_fn(x: jnp.ndarray, t: float):
        x_shape = x.shape
        dim = np.prod(x_shape[1:])
        t = jnp.expand_dims(t.reshape(-1), axis=-1)
        x = jnp.expand_dims(x, 1)  # NOTE: need leading batch dim after vmap
        t = jnp.expand_dims(t, 1)
        jac = jax.vmap(jax.jacrev(fi_fn, argnums=0))(x, t)
        jac = jac.reshape([x_shape[0], dim, dim])
        if Xi is not None:
            jac = jnp.einsum("...nd,...dm->...nm", jac, Xi)
        div = jnp.trace(jac, axis1=-1, axis2=-2)  # .reshape(x_shape[:-1])
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


class LieAlgebraGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

    @staticmethod
    def output_shape(manifold):
        return manifold.dim

    def _generators(self, x):
        return self.manifold.lie_algebra.basis  # / jnp.sqrt(2)

    def __call__(self, x, t):
        x = x.reshape((x.shape[0], self.manifold.dim, self.manifold.dim))
        fi_fn, Xi_fn = self.decomposition
        # TODO: what representation to use for NN's input?
        x_input = x.reshape((*x.shape[:-2], -1))
        # x_input = self.manifold.vee(self.manifold.log(x)) #NOTE: extremly unstable
        fi, Xi = fi_fn(x_input, t), Xi_fn(x)
        out = jnp.einsum("...i,ijk ->...jk", fi, Xi)
        # is_tangent = self.manifold.lie_algebra.belongs(out, atol=1e-3).all()
        # print(is_tangent)
        out = self.manifold.compose(x, out)
        # is_tangent = self.manifold.is_tangent(out, x, atol=1e-3).all()
        # out = self.manifold.to_tangent(out, x)
        return out.reshape((x.shape[0], -1))

    def div_split(self, x, t, hutchinson_type):
        """Returns div(X) = Xi(fi) + fi div(Xi)"""
        fi_fn, Xi_fn = self.decomposition
        Xi = Xi_fn(x)
        assert hutchinson_type == "None"
        # print("Xi", Xi.shape)
        # print("x", x.shape)
        # print("t", t.shape)
        # fi = fi_fn(x.reshape((x.shape[0], -1)), t)
        # print("fi", fi.shape)
        out = 0.0
        for k in range(self.manifold.dim):
            fn = lambda x, t: fi_fn(x, t)[..., k]
            grad_fn = jax.vmap(jax.grad(fn, argnums=0))
            grad = grad_fn(x.reshape((x.shape[0], -1)), t).reshape(x.shape)
            print("grad", grad.shape)
            grad = self.manifold.compose(self.manifold.inverse(x), grad)
            grad = self.manifold.to_tangent(grad, self.manifold.identity)
            is_tangent = self.manifold.is_tangent(grad, self.manifold.identity).all()
            print("is_tangent", is_tangent.item())
            out_k = self.manifold.metric.inner_product(grad, Xi[..., k])
            # out_k = Matrices.frobenius_product(grad, Xi[..., k])
            print("out_k", out_k[5])
            out_k2 = self.manifold.vee(grad)[..., k]
            print("out_k2", out_k2[5])
            out_k3 = jnp.inner(self.manifold.vee(grad), self.manifold.vee(Xi[..., k]))
            print("out_k3", out_k3[5])
            out += out_k

        return out


class TorusGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

        self.rot_mat = jnp.array([[0, -1], [1, 0]])

    @staticmethod
    def output_shape(manifold):
        return manifold.dim

    def _generators(self, x):
        return (
            self.rot_mat @ x.reshape((*x.shape[:-1], self.manifold.dim, 2))[..., None]
        )[..., 0]

    def __call__(self, x, t):
        weights_fn, fields_fn = self.decomposition
        weights = weights_fn(x, t)
        fields = fields_fn(x)

        return (fields * weights[..., None]).reshape(
            (*x.shape[:-1], self.manifold.dim * 2)
        )
