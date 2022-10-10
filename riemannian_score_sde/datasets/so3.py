import os

from score_sde.utils import register_dataset
from score_sde.datasets import CSVDataset

import geomstats as gs
import jax.numpy as jnp


class MatrixDataset(CSVDataset):
    def __init__(self, file, extrinsic=False, delimiter=",", skip_header=1):
        super().__init__(file, delimiter=delimiter, skip_header=skip_header)

        self.manifold = gs.geometry.hypersphere.Hypersphere(2)
        self.intrinsic_data = (
            jnp.pi * (self.data / 180.0) + jnp.array([jnp.pi / 2, jnp.pi])[None, :]
        )
        self.data = self.manifold.spherical_to_extrinsic(self.intrinsic_data)
