import os

from score_sde.utils import register_dataset
from score_sde.datasets import CSVDataset

import geomstats as gs
import jax.numpy as jnp


class SphericalDataset(CSVDataset):
    def __init__(
        self, file, batch_dims, rng, extrinsic=False, delimiter=",", skip_header=1
    ):
        super().__init__(
            file, batch_dims, rng, delimiter=delimiter, skip_header=skip_header
        )

        self.manifold = gs.geometry.hypersphere.Hypersphere(2)
        self.intrinsic_data = (
            jnp.pi * (self.data / 180.0) + jnp.array([jnp.pi / 2, jnp.pi])[None, :]
        )
        self.data = self.manifold.spherical_to_extrinsic(self.intrinsic_data)


class VolcanicErruption(SphericalDataset):
    def __init__(self, batch_dims, rng, data_dir="data"):
        super().__init__(
            os.path.join(data_dir, "volerup.csv"), batch_dims, rng, skip_header=2
        )


class Fire(SphericalDataset):
    def __init__(self, batch_dims, rng, data_dir="data"):
        super().__init__(os.path.join(data_dir, "fire.csv"), batch_dims, rng)


class Flood(SphericalDataset):
    def __init__(self, batch_dims, rng, data_dir="data"):
        super().__init__(
            os.path.join(data_dir, "flood.csv"), batch_dims, rng, skip_header=2
        )


class Earthquake(SphericalDataset):
    def __init__(self, batch_dims, rng, data_dir="data"):
        super().__init__(
            os.path.join(data_dir, "quakes_all.csv"), batch_dims, rng, skip_header=4
        )
