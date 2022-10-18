from abc import ABC, abstractmethod
from jax import numpy as jnp


class BetaSchedule(ABC):
    @abstractmethod
    def beta_t(self, t):
        pass

    @abstractmethod
    def log_mean_coeff(self, t):
        pass

    @abstractmethod
    def reverse(self):
        pass


class LinearBetaSchedule(BetaSchedule):
    def __init__(
        self,
        tf: float = 1,
        t0: float = 0,
        beta_0: float = 0.1,
        beta_f: float = 20,
    ):
        self.tf = tf
        self.t0 = t0
        self.beta_0 = beta_0
        self.beta_f = beta_f

    def log_mean_coeff(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return -0.5 * (
            0.5 * normed_t**2 * (self.beta_f - self.beta_0) + normed_t * self.beta_0
        )

    def rescale_t(self, t):
        return -2 * self.log_mean_coeff(t)

    def beta_t(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return self.beta_0 + normed_t * (self.beta_f - self.beta_0)

    def reverse(self):
        return LinearBetaSchedule(
            tf=self.t0, t0=self.tf, beta_f=self.beta_0, beta_0=self.beta_f
        )


class QuadraticBetaSchedule(BetaSchedule):
    def __init__(self, tf, t0=0, beta_0=1.0, beta_f=1.0):
        self.tf = tf
        self.t0 = t0
        self.beta_f = beta_f
        self.beta_0 = beta_0

    def beta_t(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)

        return self.beta_0 + normed_t**2 * (self.beta_f - self.beta_0)

    def rescale_t(self, t):
        return -2 * self.log_mean_coeff(t)

    def log_mean_coeff(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return -0.5 * (
            normed_t * self.beta_0 + 1 / 3 * normed_t**3 * (self.beta_f - self.beta_0)
        )

    def reverse(self):
        return QuadraticBetaSchedule(
            tf=self.t0, t0=self.tf, beta_f=self.beta_0, beta_0=self.beta_f
        )


class ConstantBetaSchedule(LinearBetaSchedule):
    def __init__(
        self,
        tf: float = 1,
        value: float = 1,
    ):
        super().__init__(tf=tf, t0=0.0, beta_0=value, beta_f=value)


class TriangleBetaSchedule(BetaSchedule):
    def __init__(
        self,
        tf: float,
        t0: float = 0,
        beta_min: float = 0.1,
        beta_max: float = 20,
        peak_t: float = 0.5,
    ):
        self.tf = tf
        self.t0 = t0
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.peak_t = peak_t

    def log_mean_coeff(self, t):
        peak_t = (self.tf + self.t0) * self.peak_t
        up_t = jnp.minimum(peak_t, t)
        up_leg = (
            -0.25 * up_t**2 * (self.beta_max - self.beta_min)
            - 0.5 * up_t * self.beta_min
        )

        down_t = jnp.maximum(t - peak_t, 0)
        down_leg = (
            0.25 * down_t**2 * (self.beta_max - self.beta_min)
            - 0.5 * down_t * self.beta_max
        )
        return down_leg + up_leg

    def beta_t(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)
        up_leg = self.beta_min + normed_t * (self.beta_max - self.beta_min)
        down_leg = self.beta_max - (normed_t - 0.5) * (self.beta_max - self.beta_min)
        return jnp.where(normed_t < 0.5, up_leg, down_leg)

    def reverse(self):
        return TriangleBetaSchedule(
            tf=self.tf,
            t0=self.t0,
            beta_max=self.beta_max,
            beta_min=self.beta_min,
            peak_t=(1.0 - self.peak_t),
        )


class ReverseBetaSchedule(BetaSchedule):
    def __init__(self, forward_schedule):

        self.forward_schedule = forward_schedule
        self.tf = forward_schedule.tf
        self.t0 = forward_schedule.t0
        self.beta_f = forward_schedule.beta_0
        self.beta_0 = forward_schedule.beta_f

    def beta_t(self, t):
        t = self.tf - t
        return self.forward_schedule.beta_t(t)

    def reverse(self):
        return self.forward_schedule
