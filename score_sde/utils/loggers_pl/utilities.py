import os
import importlib
from importlib.util import find_spec
from packaging.version import Version
import pkg_resources
from pkg_resources import DistributionNotFound
from pathlib import Path
from functools import wraps, partial
import warnings
import dataclasses
from argparse import Namespace
from collections import defaultdict, OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import yaml
from warnings import warn
import fsspec
from fsspec.implementations.local import AbstractFileSystem, LocalFileSystem


def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.
    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except ModuleNotFoundError:
        return False


def _compare_version(package: str, op: Callable, version: str, use_base_version: bool = False) -> bool:
    """Compare package version with some requirements.
    >>> _compare_version("torch", operator.ge, "0.1")
    True
    """
    try:
        pkg = importlib.import_module(package)
    except (ModuleNotFoundError, DistributionNotFound):
        return False
    try:
        if hasattr(pkg, "__version__"):
            pkg_version = Version(pkg.__version__)
        else:
            # try pkg_resources to infer version
            pkg_version = Version(pkg_resources.get_distribution(package).version)
    except TypeError:
        # this is mocked by Sphinx, so it should return True to generate all summaries
        return True
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)
    return op(pkg_version, Version(version))


_OMEGACONF_AVAILABLE = _module_available("omegaconf")
if _OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf
    from omegaconf.dictconfig import DictConfig
    from omegaconf.errors import UnsupportedValueType, ValidationError


def _warn(message: Union[str, Warning], stacklevel: int = 2, **kwargs: Any) -> None:
    if type(stacklevel) is type and issubclass(stacklevel, Warning):
        rank_zero_deprecation(
            "Support for passing the warning category positionally is deprecated in v1.6 and will be removed in v1.8"
            f" Please, use `category={stacklevel.__name__}`."
        )
        kwargs["category"] = stacklevel
        stacklevel = kwargs.pop("stacklevel", 2)
    warnings.warn(message, stacklevel=stacklevel, **kwargs)


def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on rank 0."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def _get_rank() -> int:
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank())


@rank_zero_only
def rank_zero_warn(message: Union[str, Warning], stacklevel: int = 4, **kwargs: Any) -> None:
    """Function used to log warn-level messages only on rank 0."""
    _warn(message, stacklevel=stacklevel, **kwargs)


rank_zero_deprecation = partial(rank_zero_warn, category=DeprecationWarning)


class MisconfigurationException(Exception):
    """Exception used to inform users of mis-use with PyTorch Lightning."""


class AttributeDict(Dict):
    """Extended dictionary accessible with dot notation.
    >>> ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    >>> ad.key1
    1
    >>> ad.update({'my-key': 3.14})
    >>> ad.update(new_key=42)
    >>> ad.key1 = 2
    >>> ad
    "key1":    2
    "key2":    abc
    "my-key":  3.14
    "new_key": 42
    """

    def __getattr__(self, key: str) -> Optional[Any]:
        try:
            return self[key]
        except KeyError as exp:
            raise AttributeError(f'Missing attribute "{key}"') from exp

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __repr__(self) -> str:
        if not len(self):
            return ""
        max_key_length = max(len(str(k)) for k in self)
        tmp_name = "{:" + str(max_key_length + 3) + "s} {}"
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        out = "\n".join(rows)
        return out


def _is_namedtuple(obj: object) -> bool:
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def _is_dataclass_instance(obj: object) -> bool:
    # https://docs.python.org/3/library/dataclasses.html#module-level-decorators-classes-and-functions
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def apply_to_collection(
    data: Any,
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type]]] = None,
    include_none: bool = True,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.
    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        include_none: Whether to include an element if the output of ``function`` is ``None``.
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)
    Returns:
        The resulting collection
    """
    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    elem_type = type(data)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        out = []
        for k, v in data.items():
            v = apply_to_collection(
                v, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, **kwargs
            )
            if include_none or v is not None:
                out.append((k, v))
        if isinstance(data, defaultdict):
            return elem_type(data.default_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))

    is_namedtuple = _is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple or is_sequence:
        out = []
        for d in data:
            v = apply_to_collection(
                d, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, **kwargs
            )
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple else elem_type(out)

    if _is_dataclass_instance(data):
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        fields = {}
        memo = {}
        for field in dataclasses.fields(data):
            field_value = getattr(data, field.name)
            fields[field.name] = (field_value, field.init)
            memo[id(field_value)] = field_value
        result = deepcopy(data, memo=memo)
        # apply function to each field
        for field_name, (field_value, field_init) in fields.items():
            if field_init:
                v = apply_to_collection(
                    field_value,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    include_none=include_none,
                    **kwargs,
                )
            if not field_init or (not include_none and v is None):  # retain old value
                v = getattr(data, field_name)
            try:
                setattr(result, field_name, v)
            except dataclasses.FrozenInstanceError as e:
                raise MisconfigurationException(
                    "A frozen dataclass was passed to `apply_to_collection` but this is not allowed."
                    " HINT: is your batch a frozen dataclass?"
                ) from e
        return result

    # data is neither of dtype, nor a collection
    return data


def get_filesystem(path: Union[str, Path]) -> AbstractFileSystem:
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0])
    # use local filesystem
    return LocalFileSystem()


def save_hparams_to_yaml(config_yaml, hparams: Union[dict, Namespace], use_omegaconf: bool = True) -> None:
    """
    Args:
        config_yaml: path to new YAML file
        hparams: parameters to be saved
        use_omegaconf: If omegaconf is available and ``use_omegaconf=True``,
            the hparams will be converted to ``DictConfig`` if possible.
    """
    fs = get_filesystem(config_yaml)
    if not fs.isdir(os.path.dirname(config_yaml)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(config_yaml)}.")

    # convert Namespace or AD to dict
    if isinstance(hparams, Namespace):
        hparams = vars(hparams)
    elif isinstance(hparams, AttributeDict):
        hparams = dict(hparams)

    # saving with OmegaConf objects
    if _OMEGACONF_AVAILABLE and use_omegaconf:
        # deepcopy: hparams from user shouldn't be resolved
        hparams = deepcopy(hparams)
        hparams = apply_to_collection(hparams, DictConfig, OmegaConf.to_container, resolve=True)
        with fs.open(config_yaml, "w", encoding="utf-8") as fp:
            try:
                OmegaConf.save(hparams, fp)
                return
            except (UnsupportedValueType, ValidationError):
                pass

    if not isinstance(hparams, dict):
        raise TypeError("hparams must be dictionary")

    hparams_allowed = {}
    # drop paramaters which contain some strange datatypes as fsspec
    for k, v in hparams.items():
        try:
            v = v.name if isinstance(v, Enum) else v
            yaml.dump(v)
        except TypeError:
            warn(f"Skipping '{k}' parameter because it is not possible to safely dump to YAML.")
            hparams[k] = type(v).__name__
        else:
            hparams_allowed[k] = v

    # saving the standard way
    with fs.open(config_yaml, "w", newline="") as fp:
        yaml.dump(hparams_allowed, fp)
