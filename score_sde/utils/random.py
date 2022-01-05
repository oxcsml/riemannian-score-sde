import numpy as np
import jax.random as jr


class GlobalRNG:
    def __init__(self, seed: int = np.random.randint(2147483647)):
        self.key = jr.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        (ret_key, self.key) = jr.split(self.key)
        return ret_key
