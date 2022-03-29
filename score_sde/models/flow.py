from functools import partial
from typing import Sequence

import jax
import numpy as np
import jax.numpy as jnp
from scipy import integrate

from score_sde.sde import SDE
from score_sde.models.model import get_score_fn
from score_sde.sampling import get_pc_sampler
from score_sde.utils import to_flattened_numpy, from_flattened_numpy
from score_sde.utils import (
    ParametrisedScoreFunction,
    get_exact_div_fn,
    get_estimate_div_fn,
)
from score_sde.ode import odeint


def get_div_fn(drift_fn, hutchinson_type: str = "None"):
    """Pmapped divergence of the drift function."""
    if hutchinson_type == "None":
        return lambda x, t, z, eps: get_exact_div_fn(drift_fn)(x, t, z)
    else:
        return lambda x, t, z, eps: get_estimate_div_fn(drift_fn)(x, t, z, eps)


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


def get_sde_drift_from_fn(
    sde: SDE,
    model: ParametrisedScoreFunction,
    params,
    states,
):
    def drift_fn(x: jnp.ndarray, t: float, z: jnp.ndarray) -> jnp.ndarray:
        """The drift function of the reverse-time SDE."""
        score_fn = get_score_fn(
            sde,
            model,
            params,
            states,
            train=False,
        )
        pode = sde.probability_ode(score_fn)
        return pode.coefficients(x, t, z)[0]

    return drift_fn


def get_ode_drift_fn(model, params, states):
    def drift_fn(x: jnp.ndarray, t: float, z: jnp.ndarray) -> jnp.ndarray:
        model_out, _ = model.apply(params, states, None, x=x, t=t, z=z)
        return model_out

    return drift_fn


def get_moser_drift_fn(base, model, params, states):
    def drift_fn(x: jnp.ndarray, t: float, z: jnp.ndarray) -> jnp.ndarray:
        t = t.reshape(*x.shape[:-1], 1)
        u_fn = lambda x, t: model.apply(params, states, None, x=x, t=t, z=z)[0]
        t0 = jnp.zeros_like(t)
        u = u_fn(x, t0)
        nu = jnp.exp(base.log_prob(x)).reshape(*x.shape[:-1], 1)
        div_u = get_div_fn(u_fn)(x, t0, None).reshape(*x.shape[:-1], 1)
        mu_plus = jnp.maximum(1e-5, nu - div_u)  # TODO: 0. or epsilon?
        # out = -u / (nu - (1 - t) * div_u)
        out = -u / (t * nu + (1 - t) * mu_plus)  # data -> base
        # out = u / ((1 - t) * nu + t * mu_plus)  # base -> data
        return out

    return drift_fn


class PushForward:
    """ A density estimator able to evaluate log_prob and generate samples.
    Requires specifying a base distribution.
    """

    def __init__(self, manifold, flow, base):
        self.flow = flow  # NOTE: Convention is that flow: data -> base
        self.base = base

    def __repr__(self):
        return "PushForward: base:{} flow:{}".format(self.base, self.flow)

    def get_log_prob(self, model_w_dicts, train=False):
        def log_prob(rng, y, **kwargs):
            # NOTE: Can/should we jit flow?
            flow = self.flow.get_forward(model_w_dicts, train)
            x, inv_logdets = flow(rng, y, **kwargs)  # NOTE: flow is not reversed
            log_prob = self.base.log_prob(x).reshape(-1)
            log_prob += inv_logdets
            return jnp.clip(log_prob, -1e38, 1e38)
        return log_prob

    def get_sample(self, model_w_dicts, train=False):
        def sample(rng, shape, **kwargs):
            x = self.base.sample(rng, shape)
            flow = self.flow.get_forward(model_w_dicts, train, **kwargs)
            y, inv_logdets = flow(rng, x, reverse=True)  # NOTE: flow is reversed
            # log_prob = self.base.log_prob(x)
            # log_prob += inv_logdets.reshape(log_prob.shape)
            return y#, jnp.clip(log_prob, -1e38, 1e38)
        return sample


class SDEPushForward(PushForward):
    def __init__(self, manifold, flow):
        self.manifold = manifold
        self.sde = flow
        flow = CNF(
            t0=self.sde.t0, tf=self.sde.tf, hutchinson_type='None',
            get_drift_fn =  partial(get_sde_drift_from_fn, self.sde)
        )
        super(SDEPushForward, self).__init__(manifold, flow, self.sde.limiting)

    def get_sample(self, model_w_dicts, train=False, diffeq="sde"):
        if diffeq == "ode":
            return super().get_sample(model_w_dicts, train)
        elif diffeq == "sde":
            def sample(rng, shape, z, **kwargs):
                x = self.base.sample(rng, shape)
                score_fn = get_score_fn(self.sde, *model_w_dicts)
                score_fn = partial(score_fn, z=z)
                sampler = get_pc_sampler(
                    self.sde.reverse(score_fn), **kwargs)
                sampler = jax.jit(sampler)
                return sampler(rng, x)
        else:
            raise ValueError(diffeq)
        return sample


class MoserFlow(PushForward):
    def __init__(self, manifold, flow, base):
        flow.get_drift_fn = partial(get_moser_drift_fn, base)
        super(MoserFlow, self).__init__(manifold, flow, base)

    def get_log_prob(self, model_w_dicts, train=False):
        """Use closed-form formula since faster than solving ODE"""
        def log_prob(rng, y, **kwargs):
            x = y
            drift_fn = get_ode_drift_fn(*model_w_dicts)
            div_fn = get_div_fn(drift_fn, hutchinson_type="None")
            t0 = jnp.zeros((*x.shape[:-1],))
            div_u = div_fn(x, t0, None)
            base_prob = jnp.exp(self.base.log_prob(x)).reshape(-1)
            log_prob = jnp.log(jnp.maximum(1e-5, base_prob - div_u))
            return jnp.clip(log_prob, -1e38, 1e38)
        return log_prob


class ReverseWrapper:
    def __init__(self, module, tf):
        self.module = module
        self.tf = tf

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, *args, **kwargs):
        states = self.module(x, self.tf - t, *args, **kwargs)
        return tuple([-states[0], *states[1:]])


class CNF:
    def __init__(
        self,
        t0: float,
        tf: float,
        hutchinson_type: str = "Rademacher",
        rtol: str = 1e-5,
        atol: str = 1e-5,
        backend: str = "jax",
        get_drift_fn = get_ode_drift_fn,
        **kwargs
    ):
        self.get_drift_fn = get_drift_fn
        self.t0 = t0
        self.tf = tf
        self.ode_kwargs = dict(atol=atol, rtol=rtol)
        self.test_ode_kwargs = dict(atol=1e-5, rtol=1e-5)
        self.hutchinson_type = hutchinson_type
        self.backend = backend

    def get_forward(self, model_w_dicts, train, **kwargs):
        model, params, states = model_w_dicts
        def forward(rng, data, z=None, t0=None, tf=None, reverse=False):
            hutchinson_type = self.hutchinson_type if train else "None"

            rng, step_rng = jax.random.split(rng)
            shape = data.shape
            epsilon = div_noise(step_rng, shape, hutchinson_type)
            t0 = self.t0 if t0 is None else t0
            tf = self.tf if tf is None else tf
            eps = kwargs.get("eps", 1e-3)
            ts = jnp.array([t0 + eps, tf])
            ode_kwargs = self.ode_kwargs if train else self.test_ode_kwargs

            drift_fn = self.get_drift_fn(model, params, states)

            ############## scipy.integrate #############
            if self.backend == "scipy": # or not train: # TODO: Issue??
                drift_fn = jax.jit(self.get_drift_fn(model, params, states))
                div_fn = jax.jit(get_div_fn(drift_fn, hutchinson_type))

                def ode_func(t: float, x: jnp.ndarray) -> np.array:
                    sample = from_flattened_numpy(x[: -shape[0]], shape)
                    vec_t = jnp.ones((sample.shape[0],)) * t
                    drift = to_flattened_numpy(drift_fn(sample, vec_t))
                    logp_grad = to_flattened_numpy(div_fn(sample, vec_t, epsilon))
                    return np.concatenate([drift, logp_grad], axis=0)

                init = jnp.concatenate(
                    [to_flattened_numpy(data), np.zeros((shape[0],))], axis=0
                )
                if reverse:
                    raise NotImplementedError("Reverse does not work w scipy")
                solution = integrate.solve_ivp(
                    ode_func, ts, init, **ode_kwargs, method="RK45"
                )

                nfe = solution.nfev
                zp = jnp.asarray(solution.y[:, -1])
                z = from_flattened_numpy(zp[: -shape[0]], shape)
                delta_logp = zp[-shape[0] :]  # .reshape((shape[0], shape[1]))

            ################ .ode.odeint ###############
            elif self.backend == "jax":
                def ode_func(x: jnp.ndarray, t: jnp.ndarray, params, states) -> np.array:
                    sample = x[:, :-1]#.reshape(shape)
                    vec_t = jnp.ones((sample.shape[0],)) * t
                    drift_fn = self.get_drift_fn(model, params, states)
                    drift_fn = jax.jit(drift_fn)  #TODO: useless?
                    drift = drift_fn(sample, vec_t, z)
                    div_fn = get_div_fn(drift_fn, hutchinson_type)
                    div_fn = jax.jit(div_fn)  #TODO: useless?
                    logp_grad = div_fn(sample, vec_t, z, epsilon).reshape([shape[0], 1])
                    return jnp.concatenate([drift, logp_grad], axis=1)

                data = data.reshape(shape[0], -1)
                init = jnp.concatenate([data, np.zeros((shape[0], 1))], axis=1)
                ode_func = ReverseWrapper(ode_func, tf) if reverse else ode_func
                y, nfe = odeint(ode_func, init, ts, params, states, **ode_kwargs)
                z = y[-1, ..., :-1].reshape(shape)
                delta_logp = y[-1, ..., -1]
            else:
                raise ValueError(f"{self.backend} is not a valid option.")
            print(f"nfe: {nfe}")
            return z, delta_logp

        return forward
