from dataclasses import dataclass

import jax
import haiku as hk
import jax.numpy as jnp

from .layers import get_activation
from score_sde.utils import register_model

@register_model
@dataclass
class MLP:
    hidden_shapes: list
    output_shape: list
    act: str

    def __call__(self, x):
        for hs in self.hidden_shapes:
            x = hk.Linear(output_size=hs)(x)
            x = get_activation(self.act)(x)

        x = hk.Linear(output_size=self.output_shape)(x)

        return x