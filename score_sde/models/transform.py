import abc
import functools
import math
import operator
import math
import jax.numpy as jnp
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.euclidean import Euclidean


class Transform(abc.ABC):
    def __init__(self, domain, codomain):
        self.domain = domain
        self.codomain = codomain

    @abc.abstractmethod
    def __call__(self, x):
        """Computes the transform `x => y`."""

    @abc.abstractmethod
    def inv(self, y):
        """Inverts the transform `y => x`."""

    @abc.abstractmethod
    def log_abs_det_jacobian(self, x, y):
        """ Computes the log det jacobian `log |dy/dx|` given input and output."""


class ComposeTransform(Transform):
    def __init__(self, parts):
        assert len(parts) > 0
        # NOTE: Could check constraints on domains and codomains
        super().__init__(parts[0].domain, parts[-1].codomain)
        self.parts = parts

    def __call__(self, x):
        print('call')
        for part in self.parts:
            print(part)
            print('x', x[0])
            x = part(x)
            print('y', x[0])
            print('x prime', part.inv(x)[0])
        return x

    def inv(self, y):
        print('inv')
        print(y[0])
        for part in self.parts[::-1]:
            print(part)
            y = part.inv(y)
            print(y[0])
        return y

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
    def __init__(self, manifold, **kwargs):
        super().__init__(manifold, manifold)

    def __call__(self, x):
        return x

    def inv(self, y):
        return y

    def log_abs_det_jacobian(self, x, y):
        return jnp.zeros((x.shape[:-1]))


class ExpMap(Transform):
    def __init__(self, manifold, base_point=None, **kwargs):
        super().__init__(Euclidean(manifold.dim), manifold)
        self.manifold = manifold
        self.base_point = manifold.identity if base_point is None else base_point

    def __call__(self, x):
        return self.manifold.metric.exp(x, base_point=self.base_point)

    def inv(self, y):
        return self.manifold.metric.log(y, base_point=self.base_point)

    def log_abs_det_jacobian(self, x, y):
        return self.manifold.metric.logdetexp(x, y)


class TanhExpMap(ComposeTransform):
    def __init__(self, manifold, base_point=None, **kwargs):
        injectivity_radius = manifold.injectivity_radius
        if jnp.isposinf(injectivity_radius):
            parts = []
        else:
            parts = [RadialTanhTransform(0.99 * injectivity_radius, manifold.dim)]
        exp_transform = ExpMap(manifold, base_point)
        self.base_point = exp_transform.base_point
        parts.append(exp_transform)
        super().__init__(parts)


class InvStereographic(Transform):
    def __init__(self, manifold, base_point=None, **kwargs):
        assert isinstance(manifold, Hypersphere)
        super().__init__(Euclidean(manifold.dim), manifold)
        self.manifold = manifold
        assert base_point is None or base_point == manifold.identity
        self.base_point = manifold.identity

    def __call__(self, x):
        return self.manifold.inv_stereographic_projection(x)

    def inv(self, y):
        return self.manifold.stereographic_projection(y)

    def log_abs_det_jacobian(self, x, y):
        return self.manifold.inv_stereographic_projection_logdet(x)


class RadialTanhTransform(Transform):
    r"""
    from: https://github.com/pimdh/relie/blob/master/relie/flow/radial_tanh_transform.py
    Transform R^d of radius (0, inf) to (0, R)
    Uses the fact that tanh is linear near 0.
    """
    # domain = constraints.real
    # codomain = constraints.real
    # bijective = True
    # event_dim = 1

    def __init__(self, radius, dim):
        super().__init__(Euclidean(dim), Euclidean(dim))
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