import math

import jax.numpy as jnp

from geomstats.geometry.lie_group import MatrixLieGroup
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.euclidean import Euclidean
from geomstats import algebra_utils as utils

from score_sde.models import Transform, ComposeTransform


def get_likelihood_fn_w_transform(likelihood_fn, transform):
    def log_prob(x, context=None):
        y = transform.inv(x)
        logp, nfe = likelihood_fn(y, context=context)
        log_abs_det_jacobian = transform.log_abs_det_jacobian(y, x)
        logp -= log_abs_det_jacobian
        return logp, nfe

    return log_prob


class ExpMap(Transform):
    def __init__(self, manifold, base_point=None, **kwargs):
        super().__init__(Euclidean(manifold.dim), manifold)
        self.manifold = manifold
        self.base_point = manifold.identity if base_point is None else base_point
        if (self.base_point == manifold.identity).all() and isinstance(
            manifold, MatrixLieGroup
        ):
            self.forward = lambda x: manifold.exp_from_identity(x)
            self.inverse = lambda y: manifold.log_from_identity(y)
        else:
            # self.manifold.metric.exp(x, base_point=self.base_point)
            self.forward = lambda x: manifold.exp(x, base_point=self.base_point)
            self.inverse = lambda y: manifold.log(y, base_point=self.base_point)

    def __call__(self, x):
        x = self.manifold.hat(x)
        return self.forward(x)

    def inv(self, y):
        x = self.inverse(y)
        return self.manifold.vee(x)

    def log_abs_det_jacobian(self, x, y):
        # TODO: factor
        if isinstance(self.manifold, MatrixLieGroup):
            return self.manifold.logdetexp(x, y)
        else:
            return self.manifold.logdetexp(self.base_point, y)


class TanhExpMap(ComposeTransform):
    def __init__(self, manifold, base_point=None, radius=None, **kwargs):
        if radius is None:
            radius = manifold.injectivity_radius
        if jnp.isposinf(radius):
            parts = []
        else:
            parts = [RadialTanhTransform(radius, manifold.dim)]
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

    def __init__(self, radius, dim):
        super().__init__(Euclidean(dim), Euclidean(dim))
        self.radius = radius

    def __call__(self, x):
        """x -> tanh(||x||) x / ||x|| * R"""
        # x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        # mask = x_norm > 1e-8
        # x_norm = jnp.where(mask, x_norm, jnp.ones_like(x_norm))
        # return jnp.where(
        #     mask, jnp.tanh(x_norm) * x / x_norm * self.radius, x * self.radius
        # )
        x_sq_norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
        tanh_ratio = utils.taylor_exp_even_func(x_sq_norm, utils.tanh_close_0, order=5)
        return tanh_ratio * x * self.radius

    def inv(self, y):
        """
        y -> arctanh(||y|| / R) y / ||y||
        y -> arctanh(||y|| / R) y / (||y|| / R) / R
        """
        # y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
        # mask = y_norm > 1e-8
        # y_norm = jnp.where(mask, y_norm, jnp.ones_like(y_norm))
        # return jnp.where(
        #     mask, jnp.arctanh(y_norm / self.radius) * y / y_norm, y / self.radius
        # )

        y_sq_norm = jnp.sum(jnp.square(y), axis=-1, keepdims=True)
        y_sq_norm = y_sq_norm / (self.radius**2)
        # y_sq_norm = jnp.clip(y_sq_norm, a_max=1)
        y_sq_norm = jnp.clip(y_sq_norm, a_max=1 - 1e-7)
        arctanh = utils.taylor_exp_even_func(
            y_sq_norm, utils.arctanh_card_close_0, order=5
        )
        return arctanh * y / self.radius

    def log_abs_det_jacobian(self, x, y):
        """
        computation similar to exp map in https://arxiv.org/abs/1902.02992
        x -> dim * log R + (dim - 1) * log(tanh(r)/r) + log1p(- tanh(r^2))
        :param x: Tensor
        :param y: Tensor
        :return: Tensor
        """
        x_sq_norm = jnp.sum(jnp.square(x), axis=-1)
        x_norm = jnp.sqrt(x_sq_norm)
        dim = x.shape[-1]
        # tanh = jnp.tanh(x_norm)
        # term1 = -jnp.log(x_norm / tanh)
        # term1 = -jnp.log(
        #     utils.taylor_exp_even_func(x_sq_norm, utils.inv_tanh_close_0, order=5)
        # )
        # term2 = jnp.log1p(-tanh ** 2)
        # return jnp.where(x_norm > 1e-8, out, log_radius)
        term1 = utils.taylor_exp_even_func(x_sq_norm, utils.log_tanh_close_0, order=4)
        term2 = utils.taylor_exp_even_func(
            x_sq_norm, utils.log1p_m_tanh_sq_close_0, order=5
        )

        log_radius = math.log(self.radius) * jnp.ones_like(x_norm)
        out = dim * log_radius + (dim - 1) * term1 + term2
        return out
