import abc
import functools
import operator

import jax.numpy as jnp


def get_likelihood_fn_w_transform(likelihood_fn, transform):
    def log_prob(x, context=None):
        y = transform.inv(x)
        logp, nfe = likelihood_fn(y, context=context)
        log_abs_det_jacobian = transform.log_abs_det_jacobian(y, x)
        logp -= log_abs_det_jacobian
        return logp, nfe

    return log_prob


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
        """Computes the log det jacobian `log |dy/dx|` given input and output."""


class ComposeTransform(Transform):
    def __init__(self, parts):
        assert len(parts) > 0
        # NOTE: Could check constraints on domains and codomains
        super().__init__(parts[0].domain, parts[-1].codomain)
        self.parts = parts

    def __call__(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def inv(self, y):
        for part in self.parts[::-1]:
            y = part.inv(y)
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
    def __init__(self, domain, **kwargs):
        super().__init__(domain, domain)

    def __call__(self, x):
        return x

    def inv(self, y):
        return y

    def log_abs_det_jacobian(self, x, y):
        return jnp.zeros((x.shape[0]))
