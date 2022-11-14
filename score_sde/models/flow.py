from functools import partial
from typing import Sequence

import jax
import numpy as np
import jax.numpy as jnp
from score_sde.sde import SDE
from score_sde.models.transform import Id
from score_sde.utils import (
    ParametrisedScoreFunction,
    get_exact_div_fn,
    get_estimate_div_fn,
)

from score_sde.ode import odeint
from score_sde.sampling import get_pc_sampler


def get_div_fn(drift_fn, hutchinson_type: str = "None"):
    """Euclidean divergence of the drift function."""
    if hutchinson_type == "None":
        return lambda y, t, context, eps: get_exact_div_fn(drift_fn)(y, t, context)
    else:
        return lambda y, t, context, eps: get_estimate_div_fn(drift_fn)(
            y, t, context, eps
        )


def get_riemannian_div_fn(func, hutchinson_type: str = "None", manifold=None):
    """divergence of the drift function.
    if M is submersion with euclidean ambient metric: div = div_E
    else (in a char) div f = 1/sqrt(g) \sum_i \partial_i(sqrt(g) f_i)
    """
    sqrt_g = (
        lambda x: 1.0
        if manifold is None or not hasattr(manifold.metric, "lambda_x")
        else manifold.metric.lambda_x(x)
    )
    drift_fn = lambda y, t, context: sqrt_g(y) * func(y, t, context)
    div_fn = get_div_fn(drift_fn, hutchinson_type)
    return lambda y, t, context, eps: div_fn(y, t, context, eps) / sqrt_g(y)


def div_noise(
    rng: jax.random.KeyArray, shape: Sequence[int], hutchinson_type: str
) -> jnp.ndarray:
    """Sample noise for the hutchinson estimator."""
    if hutchinson_type == "Gaussian":
        epsilon = jax.random.normal(rng, shape)
    elif hutchinson_type == "Rademacher":
        epsilon = (
            jax.random.randint(rng, shape, minval=0, maxval=2).astype(jnp.float32) * 2
            - 1
        )
    elif hutchinson_type == "None":
        epsilon = None
    else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
    return epsilon


## Drift functions for ODE solver


def get_sde_drift_from_fn(sde: SDE, model: ParametrisedScoreFunction, params, states):
    def drift_fn(y: jnp.ndarray, t: float, context: jnp.ndarray) -> jnp.ndarray:
        """The drift function of the reverse-time SDE."""
        score_fn = sde.reparametrise_score_fn(model, params, states, False)
        pode = sde.probability_ode(score_fn)
        return pode.coefficients(y, t, context)[0]

    return drift_fn


def get_ode_drift_fn(model, params, states):
    def drift_fn(y: jnp.ndarray, t: float, context: jnp.ndarray) -> jnp.ndarray:
        model_out, _ = model.apply(params, states, None, y=y, t=t, context=context)
        return model_out

    return drift_fn


def get_moser_drift_fn(base, eps, model, params, states):
    def drift_fn(y: jnp.ndarray, t: float, context: jnp.ndarray) -> jnp.ndarray:
        t = t.reshape(*y.shape[:-1], 1)
        u_fn = lambda y, t, context: model.apply(
            params, states, None, y=y, t=t, context=context
        )[0]
        t0 = jnp.zeros_like(t)
        u = u_fn(y, t0, context)
        nu = jnp.exp(base.log_prob(y)).reshape(*y.shape[:-1], 1)
        div_fn = get_riemannian_div_fn(u_fn, "None", base.manifold)
        div_u = div_fn(y, t0, context, None).reshape(*y.shape[:-1], 1)
        mu_plus = jnp.maximum(eps, nu - div_u)
        out = -u / (t * nu + (1 - t) * mu_plus)  # data -> base
        return out

    return drift_fn


## Pushforwards probability measures


class PushForward:
    """
    A density estimator able to evaluate log_prob and generate samples.
    Requires specifying a base distribution.
    Generative model: z -> y -> x \in M
    """

    def __init__(self, flow, base, transform=Id):
        self.flow = flow  # NOTE: Convention is that flow: data -> base
        self.base = base
        self.transform = transform

    def __repr__(self):
        return "PushForward: base:{} flow:{}".format(self.base, self.flow)

    def get_log_prob(self, model_w_dicts, train=False, transform=True, **kwargs):
        def log_prob(x, context=None, rng=None):
            y = self.transform.inv(x) if transform else x
            tf = kwargs.pop("tf") if "tf" in kwargs else 1.0
            t0 = kwargs.pop("t0") if "t0" in kwargs else 0.0

            flow = self.flow.get_forward(model_w_dicts, train, augmented=True, **kwargs)
            z, inv_logdets, nfe = flow(
                y, context, rng=rng, tf=tf, t0=t0
            )  # NOTE: flow is not reversed
            log_prob = self.base.log_prob(z).reshape(-1)
            log_prob += inv_logdets
            if transform:
                log_prob -= self.transform.log_abs_det_jacobian(y, x)
            return jnp.clip(log_prob, -1e38, 1e38), nfe

        return log_prob

    def get_sampler(
        self, model_w_dicts, train=False, reverse=True, transform=True, **kwargs
    ):
        def sample(rng, shape, context, z=None):
            z = self.base.sample(rng, shape) if z is None else z
            flow = self.flow.get_forward(model_w_dicts, train, **kwargs)
            y, nfe = flow(z, context, reverse=reverse)  # NOTE: flow is reversed
            x = self.transform(y) if transform else y
            return x

        return sample


class SDEPushForward(PushForward):
    def __init__(self, flow: SDE, base, diffeq="sde", transform=Id):
        self.sde = flow
        self.diffeq = diffeq
        flow = CNF(
            t0=self.sde.t0,
            tf=self.sde.tf,
            get_drift_fn=partial(get_sde_drift_from_fn, self.sde),
            manifold=flow.manifold,
        )
        super(SDEPushForward, self).__init__(flow, base, transform)

    def get_sampler(
        self, model_w_dicts, train=False, reverse=True, transform=True, **kwargs
    ):
        if self.diffeq == "ode":  # via probability flow
            sample = super().get_sampler(model_w_dicts, train, reverse)
        elif self.diffeq == "sde":  # via stochastic process

            def sample(rng, shape, context, z=None):
                z = self.base.sample(rng, shape) if z is None else z
                score_fn = self.sde.reparametrise_score_fn(*model_w_dicts)
                score_fn = partial(score_fn, context=context)
                sde = self.sde.reverse(score_fn) if reverse else self.sde
                sampler = get_pc_sampler(sde, **kwargs)
                sampler = jax.jit(sampler)
                y = sampler(rng, z)
                x = self.transform(y) if transform else y
                return x

        else:
            raise ValueError(self.diffeq)
        return sample


class MoserFlow(PushForward):
    """Following https://github.com/noamroze/moser_flow/blob/main/moser.py#L36"""

    def __init__(self, flow, base, eps=1e-5, diffeq=True, transform=Id):
        self.eps = eps
        self.diffeq = diffeq
        flow.get_drift_fn = partial(get_moser_drift_fn, base, self.eps)
        super(MoserFlow, self).__init__(flow, base, transform)

    def get_log_prob(self, model_w_dicts, train=False, transform=True, **kwargs):
        if self.diffeq:
            # Evaluating 'true' likelihood via solving augmented ODE as in CNFs
            return super().get_log_prob(model_w_dicts, train, **kwargs)
        else:
            # Proxy 'trick' likelihood /!\ does not yield normalised measure /!\
            def log_prob(x, context=None):
                """Use closed-form formula since faster than solving ODE"""
                y = self.transform.inv(x) if transform else x
                log_prob = self.density(y, context, model_w_dicts, "None", None)
                if transform:
                    log_prob -= self.transform.log_abs_det_jacobian(y, x)
                nfe = 0
                return jnp.clip(log_prob, -1e38, 1e38), nfe

            return log_prob

    def nu(self, x):
        return jnp.exp(self.base.log_prob(x)).reshape(-1)

    def divergence(self, x, context, model_w_dicts, hutchinson_type, rng):
        drift_fn = get_ode_drift_fn(*model_w_dicts)
        div_fn = get_riemannian_div_fn(drift_fn, hutchinson_type, self.base.manifold)
        t = jnp.zeros((x.shape[0], 1))  # since vector field is time independant
        epsilon = div_noise(rng, x.shape, hutchinson_type)
        return div_fn(x, t, context, epsilon)

    def signed_mu(self, x, *args):
        return self.nu(x) - self.divergence(x, *args)

    def mu_minus(self, x, *args):
        return self.eps - jnp.minimum(self.eps, self.signed_mu(x, *args))
        # return relu(-self.signed_mu(x, *args) + self.eps)

    def mu_plus(self, x, *args):
        return jnp.maximum(self.eps, self.signed_mu(x, *args))
        # return relu(self.signed_mu(x, *args) - self.eps) + self.eps

    def density(self, x, *args):
        return self.mu_plus(x, *args)


class ReverseAugWrapper:
    def __init__(self, module, tf):
        self.module = module
        self.tf = tf

    def __call__(
        self, y: jnp.ndarray, t: jnp.ndarray, context: jnp.ndarray, *args, **kwargs
    ):
        states = self.module(y, self.tf - t, context, *args, **kwargs)
        return jnp.concatenate([-states[..., :-1], states[..., [-1]]], axis=1)


class ReverseWrapper:
    def __init__(self, module, tf):
        self.module = module
        self.tf = tf

    def __call__(
        self, y: jnp.ndarray, t: jnp.ndarray, context: jnp.ndarray, *args, **kwargs
    ):
        states = self.module(y, self.tf - t, context, *args, **kwargs)
        return -states


class CNF:
    def __init__(
        self,
        t0: float = 0,
        tf: float = 1,
        hutchinson_type: str = "None",
        rtol: str = 1e-5,
        atol: str = 1e-5,
        get_drift_fn=get_ode_drift_fn,
        manifold=None,
        **kwargs,
    ):
        self.get_drift_fn = get_drift_fn
        self.t0 = t0
        self.tf = tf
        self.ode_kwargs = dict(atol=atol, rtol=rtol)
        self.test_ode_kwargs = dict(atol=1e-5, rtol=1e-5)
        self.hutchinson_type = hutchinson_type
        self.manifold = manifold

    def get_forward(self, model_w_dicts, train, augmented=False, **kwargs):
        model, params, states = model_w_dicts

        def forward(data, context=None, t0=None, tf=None, rng=None, reverse=False):
            hutchinson_type = self.hutchinson_type if train else "None"

            shape = data.shape
            epsilon = div_noise(rng, shape, hutchinson_type)
            t0 = self.t0 if t0 is None else t0
            tf = self.tf if tf is None else tf
            eps = kwargs.get("eps", 1e-3)
            ts = jnp.array([t0 + eps, tf])
            ode_kwargs = self.ode_kwargs if train else self.test_ode_kwargs

            if augmented:  # Also solving for the change in log-likelihood

                def ode_func(
                    y: jnp.ndarray, t: jnp.ndarray, context: jnp.ndarray, params, states
                ) -> np.array:
                    sample = y[:, :-1]
                    vec_t = jnp.ones((sample.shape[0],)) * t
                    drift_fn = self.get_drift_fn(model, params, states)
                    drift = drift_fn(sample, vec_t, context)
                    div_fn = get_riemannian_div_fn(
                        drift_fn, hutchinson_type, self.manifold
                    )
                    logp_grad = div_fn(sample, vec_t, context, epsilon).reshape(
                        [shape[0], 1]
                    )
                    return jnp.concatenate([drift, logp_grad], axis=1)

                data = data.reshape(shape[0], -1)
                init = jnp.concatenate([data, np.zeros((shape[0], 1))], axis=1)
                ode_func = ReverseAugWrapper(ode_func, tf) if reverse else ode_func
                y, nfe = odeint(
                    ode_func, init, ts, context, params, states, **ode_kwargs
                )
                z = y[-1, ..., :-1].reshape(shape)
                delta_logp = y[-1, ..., -1]
                return z, delta_logp, nfe
            else:

                def ode_func(
                    y: jnp.ndarray, t: jnp.ndarray, context: jnp.ndarray, params, states
                ) -> np.array:
                    sample = y
                    vec_t = jnp.ones((sample.shape[0],)) * t
                    drift_fn = self.get_drift_fn(model, params, states)
                    drift = drift_fn(sample, vec_t, context)
                    return drift

                data = data.reshape(shape[0], -1)
                init = data
                ode_func = ReverseWrapper(ode_func, tf) if reverse else ode_func
                y, nfe = odeint(
                    ode_func, init, ts, context, params, states, **ode_kwargs
                )
                z = y[-1].reshape(shape)
                return z, nfe

        return forward
