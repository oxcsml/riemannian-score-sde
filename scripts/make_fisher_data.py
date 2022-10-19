import setGPU
import numpy as np
from pathlib import Path
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"

import jax.numpy as jnp
from riemannian_score_sde.datasets import Langevin
from geomstats.geometry.special_orthogonal import (
    SpecialOrthogonal,
    _SpecialOrthogonal3Vectors,
)


if __name__ == "__main__":
    manifold = SpecialOrthogonal(n=3, point_type="matrix")
    batch_size = 512
    K = 4
    scale = 2
    dir_path = f"{os.getcwd()}/data/so3_langevin/K_{K}_s_{scale}"

    params = dict(
        scale=scale,
        K=K,
        batch_dims=[batch_size],
        manifold=manifold,
        seed=0,
        conditional=True,
    )
    dataset = Langevin(**params)

    # N = batch_size * 100
    N = batch_size * 10
    Xs = []
    ks = []
    for n in range(N // batch_size):
        print(n, N // batch_size)
        X, k = next(dataset)
        Xs.append(X)
        ks.append(k)
    Xs = jnp.concatenate(Xs, axis=0)
    ks = jnp.concatenate(ks, axis=0)
    Xs = Xs.reshape(N, -1)
    ks = ks.reshape(N, -1)

    Path(dir_path).mkdir(parents=True, exist_ok=True)
    np.savetxt(f"{dir_path}/X.csv", Xs, delimiter=",")
    np.savetxt(f"{dir_path}/k.csv", ks, delimiter=",")
