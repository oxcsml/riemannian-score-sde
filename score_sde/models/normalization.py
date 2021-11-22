import functools

import haiku as hk

from score_sde.utils import register_category

_get_normalization, register_normalization = register_category("normalization")


def get_normalization(normalization, conditional=False, num_classes=None):
    norm = _get_normalization((normalization, conditional))
    if num_classes is not None:
        norm = functools.partial(norm, num_classes=num_classes)
    return norm


class VarianceNorm2d(hk.Module):
    """Variance normalization for images."""

    bias: bool = False

    @staticmethod
    def scale_init(shape, dtype):
        normal_init = hk.initializers.RandomNormal(0.02)
        return normal_init(shape, dtype=dtype) + 1.0

    def __call__(self, x):
        variance = jnp.var(x, axis=(1, 2), keepdims=True)
        h = x / jnp.sqrt(variance + 1e-5)

        scale = hk.get_parameter(
            "scale",
            shape=(1, 1, 1, x.shape[-1]),
            dtype=x.dtype,
            init=VarianceNorm2d.scale_init,
        )

        h = h * scale
        if self.bias:
            bias = hk.get_parameter(
                "bias",
                shape=(1, 1, 1, x.shape[-1]),
                dtype=x.dtype,
                init=hk.initializers.Constant(0.0),
            )
            h = h + bias

        return h


class InstanceNorm2d(hk.Module):
    """Instance normalization for images."""

    bias: bool = True

    def __call__(self, x):
        mean = jnp.mean(x, axis=(1, 2), keepdims=True)
        variance = jnp.var(x, axis=(1, 2), keepdims=True)
        h = (x - mean) / jnp.sqrt(variance + 1e-5)

        scale = hk.get_parameter(
            "scale",
            shape=(1, 1, 1, x.shape[-1]),
            dtype=x.dtype,
            init=hk.initializers.Constant(1.0),
        )

        h = h * scale

        if self.bias:
            bias = hk.get_parameter(
                "bias",
                shape=(1, 1, 1, x.shape[-1]),
                dtype=x.dtype,
                init=hk.initializers.Constant(0.0),
            )
            h = h + bias

        return h


class InstanceNorm2dPlus(hk.Module):
    """InstanceNorm++ as proposed in the original NCSN paper."""

    bias: bool = True

    @staticmethod
    def scale_init(shape, dtype):
        normal_init = hk.initializers.RandomNormal(0.02)
        return normal_init(shape, dtype=dtype) + 1.0

    def __call__(self, x):
        means = jnp.mean(x, axis=(1, 2))
        m = jnp.mean(means, axis=-1, keepdims=True)
        v = jnp.var(means, axis=-1, keepdims=True)
        means_plus = (means - m) / jnp.sqrt(v + 1e-5)

        h = (x - means[:, None, None, :]) / jnp.sqrt(
            jnp.var(x, axis=(1, 2), keepdims=True) + 1e-5
        )

        alpha = hk.get_parameter(
            "alpha",
            shape=(1, 1, 1, x.shape[-1]),
            dtype=x.dtype,
            init=InstanceNorm2dPlus.scale_init,
        )

        gamma = hk.get_parameter(
            "gamma",
            shape=(1, 1, 1, x.shape[-1]),
            dtype=x.dtype,
            init=InstanceNorm2dPlus.scale_init,
        )

        h = h + means_plus[:, None, None, :] * alpha
        h = h * gamma
        if self.bias:
            beta = hk.get_parameter(
                "beta",
                shape=(1, 1, 1, x.shape[-1]),
                dtype=x.dtype,
                init=hk.initializers.Constant(0.0),
            )
            h = h + beta

        return h


class ConditionalInstanceNorm2dPlus(hk.Module):
    """Conditional InstanceNorm++ as in the original NCSN paper."""

    num_classes: int = 10
    bias: bool = True

    def __call__(self, x, y):
        means = jnp.mean(x, axis=(1, 2))
        m = jnp.mean(means, axis=-1, keepdims=True)
        v = jnp.var(means, axis=-1, keepdims=True)
        means_plus = (means - m) / jnp.sqrt(v + 1e-5)
        h = (x - means[:, None, None, :]) / jnp.sqrt(
            jnp.var(x, axis=(1, 2), keepdims=True) + 1e-5
        )
        normal_init = hk.initializers.RandomNormal(0.02)
        zero_init = hk.initializers.Constant(0.0)
        if self.bias:

            def init_embed(shape, dtype):
                feature_size = shape[1] // 3
                normal = normal_init((shape[0], 2 * feature_size), dtype=dtype) + 1.0
                zero = zero_init((shape[0], feature_size), dtype=dtype)
                return jnp.concatenate([normal, zero], axis=-1)

            embed = hk.Embed(
                vocab_size=self.num_classes,
                embed_dim=x.shape[-1] * 3,
                w_init=init_embed,
            )
        else:

            def init_embed(shape, dtype=jnp.float32):
                return normal_init(shape, dtype=dtype) + 1.0

            embed = hk.Embed(
                vocab_size=self.num_classes,
                embed_dim=x.shape[-1] * 2,
                w_init=init_embed,
            )

        if self.bias:
            gamma, alpha, beta = jnp.split(embed(y), 3, axis=-1)
            h = h + means_plus[:, None, None, :] * alpha[:, None, None, :]
            out = gamma[:, None, None, :] * h + beta[:, None, None, :]
        else:
            gamma, alpha = jnp.split(embed(y), 2, axis=-1)
            h = h + means_plus[:, None, None, :] * alpha[:, None, None, :]
            out = gamma[:, None, None, :] * h

        return out


register_normalization(
    ConditionalInstanceNorm2dPlus,
    name=("InstanceNorm++", True),
)
register_normalization(
    InstanceNorm2d,
    name=("InstanceNorm", False),
)
register_normalization(
    InstanceNorm2dPlus,
    name=("InstanceNorm++", False),
)
register_normalization(
    functools.partial(hk.GroupNorm, groups=32), # Compatability with flax
    name=("GroupNorm", False),
)
register_normalization(
    VarianceNorm2d,
    name=("VarianceNorm", False),
)
