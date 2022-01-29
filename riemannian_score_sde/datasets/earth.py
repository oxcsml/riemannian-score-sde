import os

from score_sde.utils import register_dataset
from score_sde.datasets import CSVDataset

import geomstats as gs
import jax.numpy as jnp


class SphericalDataset(CSVDataset):
    def __init__(self, file, extrinsic=False, delimiter=",", skip_header=1):
        super().__init__(file, delimiter=delimiter, skip_header=skip_header)

        self.manifold = gs.geometry.hypersphere.Hypersphere(2)
        self.intrinsic_data = (
            jnp.pi * (self.data / 180.0) + jnp.array([jnp.pi / 2, jnp.pi])[None, :]
        )
        self.data = self.manifold.spherical_to_extrinsic(self.intrinsic_data)


class VolcanicErruption(SphericalDataset):
    def __init__(self, data_dir="data", **kwargs):
        super().__init__(os.path.join(data_dir, "volerup.csv"), skip_header=2)


class Fire(SphericalDataset):
    def __init__(self, data_dir="data", **kwargs):
        super().__init__(os.path.join(data_dir, "fire.csv"))


class Flood(SphericalDataset):
    def __init__(self, data_dir="data", **kwargs):
        super().__init__(os.path.join(data_dir, "flood.csv"), skip_header=2)


class Earthquake(SphericalDataset):
    def __init__(self, data_dir="data", **kwargs):
        super().__init__(os.path.join(data_dir, "quakes_all.csv"), skip_header=4)
