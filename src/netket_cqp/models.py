from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

try:
    import flax.linen as nn
    import netket as nk
except ImportError:  # pragma: no cover - optional runtime dependency
    nn = None
    nk = None


def _require_flax_netket():
    if nn is None or nk is None:
        raise ImportError("flax and netket are required for NetKet-compatible models.")


if nn is not None and nk is not None:

    class FFNN(nn.Module):
        """
        NetKet tutorial FFNN: Dense -> log_cosh -> sum.
        """

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(
                features=2 * x.shape[-1],
                use_bias=True,
                param_dtype=np.complex128,
                kernel_init=nn.initializers.normal(stddev=0.01),
                bias_init=nn.initializers.normal(stddev=0.01),
            )(x)
            x = nk.nn.activation.log_cosh(x)
            return jnp.sum(x, axis=-1)


    class FMHA(nn.Module):
        """
        Factored multi-head attention from the NetKet ViT tutorial.
        """

        d_model: int
        n_heads: int
        n_patches: int
        param_dtype: object = jnp.float64

        def setup(self):
            self.v = nn.Dense(
                self.d_model,
                use_bias=False,
                kernel_init=nn.initializers.xavier_uniform(),
                param_dtype=self.param_dtype,
            )
            self.W = nn.Dense(
                self.d_model,
                use_bias=False,
                kernel_init=nn.initializers.xavier_uniform(),
                param_dtype=self.param_dtype,
            )
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches, self.n_patches),
                self.param_dtype,
            )

        def __call__(self, x):
            B, P, _ = x.shape
            d_head = self.d_model // self.n_heads
            v = self.v(x).reshape(B, P, self.n_heads, d_head)
            v = jnp.transpose(v, (0, 2, 1, 3))
            x = jnp.matmul(self.alpha[None, :, :, :], v)
            x = jnp.transpose(x, (0, 2, 1, 3)).reshape(B, P, self.d_model)
            return self.W(x)


    class EncoderBlock(nn.Module):
        """
        Transformer encoder block with factored attention.
        """

        d_model: int
        n_heads: int
        n_patches: int
        d_ff: int
        param_dtype: object = jnp.float64

        @nn.compact
        def __call__(self, x):
            h = x if self.d_model == 1 else nn.LayerNorm(param_dtype=self.param_dtype)(x)
            x = x + FMHA(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                param_dtype=self.param_dtype,
            )(h)

            h = x if self.d_model == 1 else nn.LayerNorm(param_dtype=self.param_dtype)(x)
            h = nn.Dense(self.d_ff, param_dtype=self.param_dtype)(h)
            h = jax.nn.silu(h)
            h = nn.Dense(self.d_model, param_dtype=self.param_dtype)(h)
            return x + h


    class NetKetViT(nn.Module):
        """
        Flax/NetKet-compatible ViT wavefunction for 1D CQP spin chains.
        """

        L: int
        b: int = 4
        n_layers: int = 1
        d_model: int = 32
        n_heads: int = 4
        d_ff: int = 128
        param_dtype: object = jnp.float64

        @nn.compact
        def __call__(self, spins):
            x = jnp.atleast_2d(spins)
            B = x.shape[0]
            n_patches = self.L // self.b
            x = x.reshape(B, n_patches, self.b)
            x = nn.Dense(
                self.d_model,
                kernel_init=nn.initializers.xavier_uniform(),
                param_dtype=self.param_dtype,
            )(x)

            for _ in range(self.n_layers):
                x = EncoderBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_patches=n_patches,
                    d_ff=self.d_ff,
                    param_dtype=self.param_dtype,
                )(x)

            z = jnp.sum(x, axis=1)
            if self.d_model != 1:
                z = nn.LayerNorm(param_dtype=self.param_dtype)(z)
            u = nn.Dense(
                self.d_model,
                dtype=np.complex128,
                param_dtype=np.complex128,
                kernel_init=nn.initializers.normal(stddev=0.01),
                bias_init=nn.initializers.normal(stddev=0.01),
            )(z)
            return jnp.sum(nk.nn.activation.log_cosh(u), axis=-1)

else:

    class FFNN:
        def __init__(self, *args, **kwargs):
            _require_flax_netket()


    class NetKetViT:
        def __init__(self, *args, **kwargs):
            _require_flax_netket()
