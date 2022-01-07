# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""DDPM model.

This code is the Haiku equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import functools
from dataclasses import dataclass

import jax
import haiku as hk
import jax.numpy as jnp

from .normalization import get_normalization as get_normalization

from .layers import get_activation
from .layers.ddpm import ddpm_conv3x3 as conv3x3, ResnetBlockDDPM, Downsample, Upsample
from .layers.ncsn import RefineBlock, ResidualBlock
from .layers.layers import (
    AttentionBlock,

    default_init,
    get_timestep_embedding,
)

from score_sde.utils import register_model
from .model import get_sigmas


@register_model
@dataclass
class DDPM:
    """DDPM model architecture."""

    sigma_min: float
    sigma_max: float
    num_scales: int
    channel_multiplier: list
    num_res_blocks: int
    centred_data: bool
    act: str = "swish"
    normalization: str = "GroupNorm"
    num_feature: int = 128
    attention_resolutions: int = (16,)
    dropout: float = 0.0
    resample_with_conv: bool = True
    conditional: bool = True
    scale_by_sigma: bool = False


    def __call__(self, x, labels, train=True):
        # config parsing
        act = get_activation(self.act)
        normalize = get_normalization(self.normalization)
        sigmas = get_sigmas(self.sigma_min, self.sigma_max, self.num_scales)

        nf = self.num_feature
        ch_mult = self.channel_multiplier
        num_res_blocks = self.num_res_blocks
        attn_resolutions = self.attention_resolutions
        dropout = self.dropout
        resamp_with_conv = self.resample_with_conv
        num_resolutions = len(ch_mult)

        AttnBlock = functools.partial(AttentionBlock, normalize=normalize)
        ResnetBlock = functools.partial(
            ResnetBlockDDPM, act=act, normalize=normalize, dropout=dropout
        )

        if self.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = get_timestep_embedding(timesteps, nf)
            temb = hk.Linear(nf * 4, w_init=default_init())(temb)
            temb = hk.Linear(nf * 4, w_init=default_init())(act(temb))
        else:
            temb = None

        if self.centred_data:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0, 1]
            h = 2 * x - 1.0

        # Downsampling block
        hs = [conv3x3(h, nf)]
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
                if h.shape[1] in attn_resolutions:
                    h = AttnBlock()(h)
                hs.append(h)
            if i_level != num_resolutions - 1:
                hs.append(Downsample(with_conv=resamp_with_conv)(hs[-1]))

        h = hs[-1]
        h = ResnetBlock()(h, temb, train)
        h = AttnBlock()(h)
        h = ResnetBlock()(h, temb, train)

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(
                    jnp.concatenate([h, hs.pop()], axis=-1), temb, train
                )
            if h.shape[1] in attn_resolutions:
                h = AttnBlock()(h)
            if i_level != 0:
                h = Upsample(with_conv=resamp_with_conv)(h)

        assert not hs

        h = act(normalize()(h))
        h = conv3x3(h, x.shape[-1], init_scale=0.0)

        if self.scale_by_sigma:
            # Divide the output by sigmas. Useful for training with the NCSN loss.
            # The DDPM loss scales the network output by sigma in the loss function,
            # so no need of doing it here.
            used_sigmas = sigmas[labels].reshape(
                (x.shape[0], *([1] * len(x.shape[1:])))
            )
            h = h / used_sigmas

        return h
