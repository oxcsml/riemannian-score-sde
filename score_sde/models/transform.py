import abc
import functools
import math
import operator
import math
import jax.numpy as jnp
from geomstats.geometry.hypersphere import Hypersphere


class Transform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x):
        """Computes the transform `x => y`."""

    @abc.abstractmethod
    def inv(self, x):
        """Inverts the transform `y => x`."""

    @abc.abstractmethod
    def log_abs_det_jacobian(self, x, y):
        """ Computes the log det jacobian `log |dy/dx|` given input and output."""


class ComposeTransform(Transform):
    def __init__(self, parts):
        self.parts = parts

    def __call__(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def inv(self, x):
        for part in self.parts:
            x = part.inv(x)
        return x

    def log_abs_det_jacobian(self, x, y):
        xs = [x]
        for part in self.parts[:-1]:
            xs.append(part(xs[-1]))
        xs.append(y)
        terms = []
        for part, x, y in zip(self.parts, xs[:-1], xs[1:]):
            terms.append(part.log_abs_det_jacobian(x, y))
        return functools.reduce(operator.add, terms)


class Id(Transform):
    def __init__(self, **kwargs):
        pass

    def __call__(self, x):
        return x

    def inv(self, x):
        return x

    def log_abs_det_jacobian(self, x, y):
        return jnp.zeros((x.shape[:-1]))


class ExpMap(Transform):
    def __init__(self, manifold):
        self.manifold = manifold
        self.base_point = manifold.base_point

    def __call__(self, x):
        return self.manifold.metric.exp(x, base_point=self.base_point)

    def inv(self, x):
        return self.manifold.metric.log(x, base_point=self.base_point)

    def log_abs_det_jacobian(self, x, y):
        return self.manifold.logdetexp(y, base_point=self.base_point)


class TanhExpMap(ComposeTransform):
    def __init__(self, manifold):
        injectivity_radius = manifold.injectivity_radius
        if jnp.isposinf(injectivity_radius):
            parts = []
        else:
            parts = [RadialTanhTransform(0.99 * injectivity_radius)]
        parts.append(ExpMap(manifold))
        super().__init__(parts)


class InvStereographic(Transform):
    def __init__(self, manifold):
        assert isinstance(manifold, Hypersphere)
        self.manifold = manifold

    def __call__(self, x):
        return self.manifold.inv_stereographic_projection(x)

    def inv(self, x):
        return self.manifold.stereographic_projection(x)

    def log_abs_det_jacobian(self, x, y):
        return self.manifold.inv_stereographic_projection_logdet(y)


class RadialTanhTransform:
    r"""
    from: https://github.com/pimdh/relie/blob/master/relie/flow/radial_tanh_transform.py
    Transform R^d of radius (0, inf) to (0, R)
    Uses the fact that tanh is linear near 0.
    """
    # domain = constraints.real
    # codomain = constraints.real
    # bijective = True
    # event_dim = 1

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, x):
        x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        mask = x_norm > 1e-8
        x_norm = jnp.where(mask, x_norm, jnp.ones_like(x_norm))

        return jnp.where(
            mask, jnp.tanh(x_norm) * x / x_norm * self.radius, x * self.radius
        )

    def inv(self, y):
        # org_dtype = y.dtype
        # y = y.double()
        y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
        mask = y_norm > 1e-8
        y_norm = jnp.where(mask, y_norm, jnp.ones_like(y_norm))

        return jnp.where(
            mask, jnp.arctanh(y_norm / self.radius) * y / y_norm, y / self.radius
        )#.to(org_dtype)

    def log_abs_det_jacobian(self, x, y):
        """
        Uses d tanh /dx = 1-tanh^2
        :param x: Tensor
        :param y: Tensor
        :return: Tensor
        """
        y_norm = jnp.linalg.norm(y, axis=-1)
        d = y.shape[-1]
        tanh = y_norm / self.radius
        log_dtanh = jnp.log1p(-tanh ** 2)

        log_radius = jnp.full_like(log_dtanh, math.log(self.radius))
        return d * jnp.where(y_norm > 1e-8, log_dtanh + log_radius, log_radius)