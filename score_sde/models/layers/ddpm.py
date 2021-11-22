###########################################################################
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################
from functools import partial
from typing import Any, Optional
from dataclasses import dataclass

import jax
import haiku as hk
import jax.numpy as jnp

from .layers import default_init, NIN


def ddpm_conv1x1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0):
    """1x1 convolution with DDPM initialization."""
    b_init = hk.initializers.Constant(0.0)
    output = hk.ConvND(
        num_spatial_dims=len(x.shape[2:]),
        output_channels=out_planes,
        kernel_shape=(1, 1),
        stride=(stride, stride),
        padding="SAME",
        use_bias=bias,
        rate=(dilation, dilation),
        w_init=default_init(scale=init_scale),
        b_init=b_init,
    )(x)
    return output


def ddpm_conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0):
    """3x3 convolution with DDPM initialization."""
    b_init = hk.initializers.Constant(0.0)
    output = hk.ConvND(
        num_spatial_dims=len(x.shape[2:]),
        output_channels=out_planes,
        kernel_shape=(3, 3),
        stride=(stride, stride),
        padding="SAME",
        with_bias=bias,
        rate=(dilation, dilation),
        w_init=default_init(scale=init_scale),
        b_init=b_init,
    )(x)
    return output

@dataclass
class ResnetBlockDDPM(hk.Module):
    """The ResNet Blocks used in DDPM."""

    act: Any
    normalize: Any
    out_ch: Optional[int] = None
    conv_shortcut: bool = False
    dropout: float = 0.5

    def __call__(self, x, temb=None, train=True):
        B, H, W, C = x.shape
        out_ch = self.out_ch if self.out_ch else C
        h = self.act(self.normalize()(x))
        h = ddpm_conv3x3(h, out_ch)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += hk.Linear(out_ch, w_init=default_init())(self.act(temb))[
                :, None, None, :
            ]
        h = self.act(self.normalize()(h))
        if train:
            h = hk.dropout(hk.next_rng_key(), x=h, rate=self.dropout)
        h = ddpm_conv3x3(h, out_ch, init_scale=0.0)
        if C != out_ch:
            if self.conv_shortcut:
                x = ddpm_conv3x3(x, out_ch)
            else:
                x = NIN(out_ch)(x)
        return x + h



@dataclass
class Upsample(hk.Module):
    with_conv: bool = False

    def __call__(self, x):
        B, H, W, C = x.shape
        h = jax.image.resize(x, (x.shape[0], H * 2, W * 2, C), "nearest")
        if self.with_conv:
            h = ddpm_conv3x3(h, C)
        return h


@dataclass
class Downsample(hk.Module):
    with_conv: bool = False

    def __call__(self, x):
        B, H, W, C = x.shape
        if self.with_conv:
            x = ddpm_conv3x3(x, C, stride=2)
        else:
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        assert x.shape == (B, H // 2, W // 2, C)
        return x
