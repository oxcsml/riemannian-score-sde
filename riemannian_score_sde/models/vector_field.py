import abc

import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp

from hydra.utils import instantiate
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.base import VectorSpace, EmbeddedManifold


# def get_exact_div_fn(fi_fn, Xi=None):
#     "flatten all but the last axis and compute the true divergence"

#     def div_fn(x: jnp.ndarray, t: float):
#         x_shape = x.shape
#         dim = np.prod(x_shape[1:])
#         t = jnp.expand_dims(t.reshape(-1), axis=-1)
#         x = jnp.expand_dims(x, 1)  # NOTE: need leading batch dim after vmap
#         t = jnp.expand_dims(t, 1)
#         jac = jax.vmap(jax.jacrev(fi_fn, argnums=0))(x, t)
#         jac = jac.reshape([x_shape[0], dim, dim])
#         if Xi is not None:
#             jac = jnp.einsum("...nd,...dm->...nm", jac, Xi)
#         div = jnp.trace(jac, axis1=-1, axis2=-2)
#         return div

#     return div_fn


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
        # NOTE: seems that extra projection is required for generator=eigen
        # during the ODE solve cf tests/test_lkelihood.py
        out = self.manifold.to_tangent(out, x)
        return out

    def div_generators(self, x):
        """Divergence of the generating vector fields: shape=[..., card=n]"""


class DivFreeGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

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
        return self.manifold.lie_algebra.basis

    def __call__(self, x, t):
        x = x.reshape((x.shape[0], self.manifold.dim, self.manifold.dim))
        fi_fn, Xi_fn = self.decomposition
        x_input = x.reshape((*x.shape[:-2], -1))
        fi, Xi = fi_fn(x_input, t), Xi_fn(x)
        out = jnp.einsum("...i,ijk ->...jk", fi, Xi)
        out = self.manifold.compose(x, out)
        # out = self.manifold.to_tangent(out, x)
        return out.reshape((x.shape[0], -1))


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


class CanonicalGenerator:
    def __init__(self, architecture, embedding, output_shape=None, manifold=None):
        self.net = instantiate(architecture, output_shape=output_shape)

    @staticmethod
    def output_shape(manifold):
        return manifold.dim

    def __call__(self, x, t):
        return self.net(x, t)


class ParallelTransportGenerator:
    def __init__(self, architecture, embedding, output_shape=None, manifold=None):
        self.net = instantiate(architecture, output_shape=output_shape)
        self.manifold = manifold

    @staticmethod
    def output_shape(manifold):
        # return manifold.dim
        return manifold.identity.shape[-1]

    def __call__(self, x, t):
        """
        Rescale since ||s(x, t)||^2_x = s(x, t)^t G(x) s(x, t) = \lambda(x)^2 ||s(x, t)||^2_2
        with G(x)=\lambda(x)^2 Id
        """
        tangent = self.net(x, t)
        tangent = self.manifold.metric.transpfrom0(x, tangent)
        return tangent
