"""Various sampling methods."""
from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp

from score_sde.utils import batch_mul
from score_sde.sampling import (
    get_pc_sampler,
    Predictor,
    Corrector,
    register_predictor,
    register_corrector,
)


@partial(register_predictor, name="GRW")
class EulerMaruyamaManifoldPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        shape = x.shape
        z = self.sde.manifold.random_normal_tangent(
            state=rng, base_point=x, n_samples=x.shape[0]
        )[1].reshape(shape[0], -1)
        drift, diffusion = self.sde.coefficients(x.reshape(shape[0], -1), t)
        drift = drift * dt[..., None]
        if len(diffusion.shape) > 1 and diffusion.shape[-1] == diffusion.shape[-2]:
            # if square matrix diffusion coeffs
            tangent_vector = drift + jnp.einsum(
                "...ij,...j,...->...i", diffusion, z, jnp.sqrt(jnp.abs(dt))
            )
        else:
            # if scalar diffusion coeffs (i.e. no extra dims on the diffusion)
            tangent_vector = drift + jnp.einsum(
                "...,...i,...->...i", diffusion, z, jnp.sqrt(jnp.abs(dt))
            )

        tangent_vector = tangent_vector.reshape(shape)
        x = self.sde.manifold.exp(tangent_vec=tangent_vector, base_point=x)
        return x, x


@register_corrector
class LangevinCorrector(Corrector):
    """
    dX = c \nabla \log p dt +  (2c)^1/2 dBt
    c = 1/2
    """

    def __init__(self, sde, snr, n_steps):
        raise NotImplementedError("This corrector has not been properly tested")
        super().__init__(sde, snr, n_steps)

    def update_fn(
        self, rng: jax.random.KeyArray, x: jnp.ndarray, t: float, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        shape = x.shape
        sde = self.sde
        n_steps = self.n_steps
        target_snr = self.snr
        """ timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
        alpha = sde.alphas[timestep] """
        alpha = jnp.ones_like(t)

        def loop_body(step, val):
            rng, x = val
            grad = sde.score_fn(x.reshape(shape[0], -1), t)
            rng, step_rng = jax.random.split(rng)

            noise = self.sde.manifold.random_normal_tangent(
                state=step_rng, base_point=x, n_samples=x.shape[0]
            )[1].reshape(shape[0], -1)

            grad_norm = self.sde.manifold.metric.norm(grad, x).mean()
            noise_norm = self.sde.manifold.metric.norm(noise, x).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            step_size = jnp.expand_dims(step_size, -1)

            tangent_vector = batch_mul((step_size / 2), grad)
            tangent_vector += batch_mul(jnp.sqrt(step_size), noise)
            tangent_vector = tangent_vector.reshape(shape)

            x = self.sde.manifold.exp(tangent_vec=tangent_vector, base_point=x)
            return rng, x

        _, x = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x))
        return x, x  # x_mean
