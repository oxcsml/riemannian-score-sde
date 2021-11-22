###########################################################################
# Functions below are ported over from the NCSNv1/NCSNv2 codebase:
# https://github.com/ermongroup/ncsn
# https://github.com/ermongroup/ncsnv2
###########################################################################
from typing import Any, Optional, Sequence
from dataclasses import dataclass

import haiku as hk
import jax.nn as jnn

from .layers import default_init


def ncsn_conv1x1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0):
    """1x1 convolution with PyTorch initialization. Same as NCSNv1/v2."""
    init_scale = 1e-10 if init_scale == 0 else init_scale
    w_init = hk.initializers.VarianceScaling(1 / 3 * init_scale, "fan_in", "uniform")
    kernel_shape = (1, 1) + (x.shape[-1], out_planes)
    b_init = lambda shape: w_init(kernel_shape)[0, 0, 0, :]
    output = hk.ConvND(
        num_spatial_dims=len(x.shape[2:]),
        output_channels=out_planes,
        kernel_shape=(1, 1),
        stride=(stride, stride),
        padding="SAME",
        with_bias=bias,
        rate=(dilation, dilation),
        w_init=w_init,
        b_init=b_init,
    )(x)
    return output


def ncsn_conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0):
    """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
    init_scale = 1e-10 if init_scale == 0 else init_scale
    w_init = hk.initializers.VarianceScaling(1 / 3 * init_scale, "fan_in", "uniform")
    kernel_shape = (3, 3) + (x.shape[-1], out_planes)
    b_init = lambda key, shape: w_init(key, kernel_shape)[0, 0, 0, :]
    output = hk.ConvND(
        num_spatial_dims=len(x.shape[2:]),
        output_channels=out_planes,
        kernel_shape=(3, 3),
        stride=(stride, stride),
        padding="SAME",
        with_bias=bias,
        rate=(dilation, dilation),
        w_init=w_init,
        b_init=b_init,
    )(x)
    return output


@dataclass
class CRPBlock(hk.Module):
    """CRPBlock for RefineNet. Used in NCSNv2."""

    features: int
    n_stages: int
    act: Any = jnn.relu

    def __call__(self, x):
        x = self.act(x)
        path = x
        for _ in range(self.n_stages):
            path = nn.max_pool(
                path, window_shape=(5, 5), strides=(1, 1), padding="SAME"
            )
            path = ncsn_conv3x3(path, self.features, stride=1, bias=False)
            x = path + x
        return x


@dataclass
class CondCRPBlock(hk.Module):
    """Noise-conditional CRPBlock for RefineNet. Used in NCSNv1."""

    features: int
    n_stages: int
    normalizer: Any
    act: Any = jnn.relu

    def __call__(self, x, y):
        x = self.act(x)
        path = x
        for _ in range(self.n_stages):
            path = self.normalizer()(path, y)
            path = nn.avg_pool(
                path, window_shape=(5, 5), strides=(1, 1), padding="SAME"
            )
            path = ncsn_conv3x3(path, self.features, stride=1, bias=False)
            x = path + x
        return x


@dataclass
class RCUBlock(hk.Module):
    """RCUBlock for RefineNet. Used in NCSNv2."""

    features: int
    n_blocks: int
    n_stages: int
    act: Any = jnn.relu

    def __call__(self, x):
        for _ in range(self.n_blocks):
            residual = x
            for _ in range(self.n_stages):
                x = self.act(x)
                x = ncsn_conv3x3(x, self.features, stride=1, bias=False)
            x = x + residual

        return x

@dataclass
class CondRCUBlock(hk.Module):
    """Noise-conditional RCUBlock for RefineNet. Used in NCSNv1."""

    features: int
    n_blocks: int
    n_stages: int
    normalizer: Any
    act: Any = jnn.relu

    def __call__(self, x, y):
        for _ in range(self.n_blocks):
            residual = x
            for _ in range(self.n_stages):
                x = self.normalizer()(x, y)
                x = self.act(x)
                x = ncsn_conv3x3(x, self.features, stride=1, bias=False)
            x += residual
        return x


@dataclass
class MSFBlock(hk.Module):
    """MSFBlock for RefineNet. Used in NCSNv2."""

    shape: Sequence[int]
    features: int
    interpolation: str = "bilinear"

    def __call__(self, xs):
        sums = jnp.zeros((xs[0].shape[0], *self.shape, self.features))
        for i in range(len(xs)):
            h = ncsn_conv3x3(xs[i], self.features, stride=1, bias=True)
            if self.interpolation == "bilinear":
                h = jax.image.resize(
                    h, (h.shape[0], *self.shape, h.shape[-1]), "bilinear"
                )
            elif self.interpolation == "nearest_neighbor":
                h = jax.image.resize(
                    h, (h.shape[0], *self.shape, h.shape[-1]), "nearest"
                )
            else:
                raise ValueError(f"Interpolation {self.interpolation} does not exist!")
            sums = sums + h
        return sums


@dataclass
class CondMSFBlock(hk.Module):
    """Noise-conditional MSFBlock for RefineNet. Used in NCSNv1."""

    shape: Sequence[int]
    features: int
    normalizer: Any
    interpolation: str = "bilinear"

    def __call__(self, xs, y):
        sums = jnp.zeros((xs[0].shape[0], *self.shape, self.features))
        for i in range(len(xs)):
            h = self.normalizer()(xs[i], y)
            h = ncsn_conv3x3(h, self.features, stride=1, bias=True)
            if self.interpolation == "bilinear":
                h = jax.image.resize(
                    h, (h.shape[0], *self.shape, h.shape[-1]), "bilinear"
                )
            elif self.interpolation == "nearest_neighbor":
                h = jax.image.resize(
                    h, (h.shape[0], *self.shape, h.shape[-1]), "nearest"
                )
            else:
                raise ValueError(f"Interpolation {self.interpolation} does not exist")
            sums = sums + h
        return sums


@dataclass
class RefineBlock(hk.Module):
    """RefineBlock for building NCSNv2 RefineNet."""

    output_shape: Sequence[int]
    features: int
    act: Any = jnn.relu
    interpolation: str = "bilinear"
    start: bool = False
    end: bool = False

    def __call__(self, xs):
        rcu_block = functools.partial(RCUBlock, n_blocks=2, n_stages=2, act=self.act)
        rcu_block_output = functools.partial(
            RCUBlock,
            features=self.features,
            n_blocks=3 if self.end else 1,
            n_stages=2,
            act=self.act,
        )
        hs = []
        for i in range(len(xs)):
            h = rcu_block(features=xs[i].shape[-1])(xs[i])
            hs.append(h)

        if not self.start:
            msf = functools.partial(
                MSFBlock, features=self.features, interpolation=self.interpolation
            )
            h = msf(shape=self.output_shape)(hs)
        else:
            h = hs[0]

        crp = functools.partial(
            CRPBlock, features=self.features, n_stages=2, act=self.act
        )
        h = crp()(h)
        h = rcu_block_output()(h)
        return h


@dataclass
class CondRefineBlock(hk.Module):
    """Noise-conditional RefineBlock for building NCSNv1 RefineNet."""

    output_shape: Sequence[int]
    features: int
    normalizer: Any
    act: Any = jnn.relu
    interpolation: str = "bilinear"
    start: bool = False
    end: bool = False

    def __call__(self, xs, y):
        rcu_block = functools.partial(
            CondRCUBlock,
            n_blocks=2,
            n_stages=2,
            act=self.act,
            normalizer=self.normalizer,
        )
        rcu_block_output = functools.partial(
            CondRCUBlock,
            features=self.features,
            n_blocks=3 if self.end else 1,
            n_stages=2,
            act=self.act,
            normalizer=self.normalizer,
        )
        hs = []
        for i in range(len(xs)):
            h = rcu_block(features=xs[i].shape[-1])(xs[i], y)
            hs.append(h)

        if not self.start:
            msf = functools.partial(
                CondMSFBlock,
                features=self.features,
                interpolation=self.interpolation,
                normalizer=self.normalizer,
            )
            h = msf(shape=self.output_shape)(hs, y)
        else:
            h = hs[0]

        crp = functools.partial(
            CondCRPBlock,
            features=self.features,
            n_stages=2,
            act=self.act,
            normalizer=self.normalizer,
        )
        h = crp()(h, y)
        h = rcu_block_output()(h, y)
        return h


@dataclass
class ConvMeanPool(hk.Module):
    """ConvMeanPool for building the ResNet backbone."""

    output_dim: int
    kernel_size: int = 3
    biases: bool = True

    def __call__(self, inputs):
        output = hk.ConvND(
            num_spatial_dims=len(x.shape[2:]),
            output_channels=self.output_dim,
            kernel_shape=(self.kernel_size, self.kernel_size),
            stride=(1, 1),
            padding="SAME",
            with_bias=self.biases,
        )(inputs)
        output = (
            sum(
                [
                    output[:, ::2, ::2, :],
                    output[:, 1::2, ::2, :],
                    output[:, ::2, 1::2, :],
                    output[:, 1::2, 1::2, :],
                ]
            )
            / 4.0
        )
        return output

@dataclass
class MeanPoolConv(hk.Module):
    """MeanPoolConv for building the ResNet backbone."""

    output_dim: int
    kernel_size: int = 3
    biases: bool = True

    def __call__(self, inputs):
        output = inputs
        output = (
            sum(
                [
                    output[:, ::2, ::2, :],
                    output[:, 1::2, ::2, :],
                    output[:, ::2, 1::2, :],
                    output[:, 1::2, 1::2, :],
                ]
            )
            / 4.0
        )
        output = hk.ConvND(
            num_spatial_dims=len(x.shape[2:]),
            output_channels=self.output_dim,
            kernel_shape=(self.kernel_size, self.kernel_size),
            stride=(1, 1),
            padding="SAME",
            with_bias=self.biases,
        )(output)
        return output


@dataclass
class ResidualBlock(hk.Module):
    """The residual block for defining the ResNet backbone. Used in NCSNv2."""

    output_dim: int
    normalization: Any
    resample: Optional[str] = None
    act: Any = jnn.elu
    dilation: int = 1

    def __call__(self, x):
        h = self.normalization()(x)
        h = self.act(h)
        if self.resample == "down":
            h = ncsn_conv3x3(h, h.shape[-1], dilation=self.dilation)
            h = self.normalization()(h)
            h = self.act(h)
            if self.dilation > 1:
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
                shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
            else:
                h = ConvMeanPool(output_dim=self.output_dim)(h)
                shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1)(x)
        elif self.resample is None:
            if self.dilation > 1:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
                h = self.normalization()(h)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
            else:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv1x1(x, self.output_dim)
                h = ncsn_conv3x3(h, self.output_dim)
                h = self.normalization()(h)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim)

        return h + shortcut


@dataclass
class ConditionalResidualBlock(hk.Module):
    """The noise-conditional residual block for building NCSNv1."""

    output_dim: int
    normalization: Any
    resample: Optional[str] = None
    act: Any = jnn.elu
    dilation: int = 1

    def __call__(self, x, y):
        h = self.normalization()(x, y)
        h = self.act(h)
        if self.resample == "down":
            h = ncsn_conv3x3(h, h.shape[-1], dilation=self.dilation)
            h = self.normalization(h, y)
            h = self.act(h)
            if self.dilation > 1:
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
                shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
            else:
                h = ConvMeanPool(output_dim=self.output_dim)(h)
                shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1)(x)
        elif self.resample is None:
            if self.dilation > 1:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
                h = self.normalization()(h, y)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
            else:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv1x1(x, self.output_dim)
                h = ncsn_conv3x3(h, self.output_dim)
                h = self.normalization()(h, y)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim)

        return h + shortcut
