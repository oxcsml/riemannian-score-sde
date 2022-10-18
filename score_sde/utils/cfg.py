import math
import functools

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


class NoneHydra:
    def __init__(self, *args, **kwargs):
        pass

    def __bool__(self):
        return False


# Define useful resolver for hydra config
# TODO: temp fix using replace due to double import in sweeps
OmegaConf.register_new_resolver("int", lambda x: int(x), replace=True)
OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
OmegaConf.register_new_resolver("str", lambda x: str(x), replace=True)
OmegaConf.register_new_resolver("prod", lambda x: np.prod(x), replace=True)
OmegaConf.register_new_resolver(
    "where", lambda condition, x, y: x if condition else y, replace=True
)
OmegaConf.register_new_resolver("isequal", lambda x, y: x == y, replace=True)
OmegaConf.register_new_resolver("pi", lambda x: x * math.pi, replace=True)
OmegaConf.register_new_resolver("min", min, replace=True)


def partialclass(cls, *args, **kwds):
    """Return a class instance with partial __init__
    Input:
        cls [str]: class to instantiate
    """
    cls = hydra.utils.get_class(cls)

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


def partialfunction(func, *args, **kwargs):
    return functools.partial(func, *args, **kwargs)
