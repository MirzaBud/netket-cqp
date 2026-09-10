from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

try:
    import netket as nk
except ImportError:  # pragma: no cover - optional runtime dependency
    nk = None


def _xavier_uniform(key, shape, dtype):
    fan_in, fan_out = shape[0], shape[1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, dtype=dtype, minval=-limit, maxval=limit)


def _complex_dtype(dtype):
    dtype = jnp.dtype(dtype)
    return jnp.complex64 if dtype.itemsize <= 4 else jnp.complex128


def _dense(x, params):
    y = x @ params["kernel"]
    if "bias" in params:
        y = y + params["bias"]
    return y


def _layer_norm(x, params, eps=1e-5):
    if x.shape[-1] == 1:
        return x * params["scale"] + params["bias"]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(var + eps)
    return x * params["scale"] + params["bias"]


def _complex_log_cosh(z):
    """
    Stable complex log(cosh(z)) with NetKet activation when it is available.
    """
    if nk is not None:
        return nk.nn.activation.log_cosh(z)

    x = z.real
    y = z.imag
    a = jnp.cosh(x) * jnp.cos(y)
    b = jnp.sinh(x) * jnp.sin(y)
    imag = jnp.arctan2(b, a)

    ratio = 0.5 * (jnp.cosh(2.0 * x) + jnp.cos(2.0 * y))
    real = 0.5 * jnp.log(jnp.maximum(ratio, 1e-30))
    return real + 1j * imag


@dataclass(frozen=True)
class ViTWaveFunction:
    """
    JAX/NetKet-style ViT variational wavefunction for 1D spin chains.

    The module mirrors the old PyTorch interface conceptually, but uses a
    functional parameter pytree so JAX can jit/vmap over samples.
    """

    L: int
    b: int = 4
    n_layers: int = 1
    d_model: int = 32
    n_heads: int = 4
    d_ff: int | None = None
    rbm_hidden: int | None = None
    symmetrize: bool = True
    marshall_sign: bool = False
    use_relative_alpha: bool = True
    final_layer_norm: bool = True
    amp_init_std: float = 1e-2
    phase_init_std: float = 1e-3
    dtype: object = jnp.float32

    def __post_init__(self):
        if self.L % self.b != 0:
            raise ValueError("L must be divisible by patch size b")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff is None:
            object.__setattr__(self, "d_ff", 2 * self.d_model)

    @property
    def P(self):
        return self.L // self.b

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    def init(self, key):
        """
        Initialize a parameter pytree compatible with JAX transformations.
        """
        keys = iter(jax.random.split(key, 4 + 5 * self.n_layers))
        rbm_hidden = self.rbm_hidden or self.d_model
        complex_dtype = _complex_dtype(self.dtype)
        rbm_kernel = (
            jax.random.normal(next(keys), (self.d_model, rbm_hidden), self.dtype) * self.amp_init_std
            + 1j
            * jax.random.normal(next(keys), (self.d_model, rbm_hidden), self.dtype)
            * self.phase_init_std
        ).astype(complex_dtype)
        rbm_bias = (
            jax.random.normal(next(keys), (rbm_hidden,), self.dtype) * self.amp_init_std
        ).astype(complex_dtype)

        params = {
            "patch_embed": {
                "kernel": _xavier_uniform(next(keys), (self.b, self.d_model), self.dtype),
                "bias": jnp.zeros((self.d_model,), dtype=self.dtype),
            },
            "layers": [],
            "rbm": {
                "kernel": rbm_kernel,
                "bias": rbm_bias,
            },
        }

        if self.final_layer_norm and self.d_model != 1:
            params["final_norm"] = {
                "scale": jnp.ones((self.d_model,), dtype=self.dtype),
                "bias": jnp.zeros((self.d_model,), dtype=self.dtype),
            }

        for _ in range(self.n_layers):
            layer = {
                "norm1": {
                    "scale": jnp.ones((self.d_model,), dtype=self.dtype),
                    "bias": jnp.zeros((self.d_model,), dtype=self.dtype),
                },
                "norm2": {
                    "scale": jnp.ones((self.d_model,), dtype=self.dtype),
                    "bias": jnp.zeros((self.d_model,), dtype=self.dtype),
                },
                "alpha": _xavier_uniform(
                    next(keys),
                    (self.n_heads, self.P) if self.use_relative_alpha else (self.n_heads, self.P, self.P),
                    self.dtype,
                ),
                "W_v": {"kernel": _xavier_uniform(next(keys), (self.d_model, self.d_model), self.dtype)},
                "W_o": {"kernel": _xavier_uniform(next(keys), (self.d_model, self.d_model), self.dtype)},
                "ff1": {
                    "kernel": _xavier_uniform(next(keys), (self.d_model, self.d_ff), self.dtype),
                    "bias": jnp.zeros((self.d_ff,), dtype=self.dtype),
                },
                "ff2": {
                    "kernel": _xavier_uniform(next(keys), (self.d_ff, self.d_model), self.dtype),
                    "bias": jnp.zeros((self.d_model,), dtype=self.dtype),
                },
            }
            params["layers"].append(layer)

        return params

    def _block(self, x, layer, rel_idx):
        """
        One factored attention block with position-only attention weights.
        """
        B, P, _ = x.shape
        h = _layer_norm(x, layer["norm1"])

        v = _dense(h, layer["W_v"])
        v = v.reshape(B, P, self.n_heads, self.d_head)
        v = jnp.transpose(v, (0, 2, 1, 3))

        if self.use_relative_alpha:
            alpha = layer["alpha"][:, rel_idx]
        else:
            alpha = layer["alpha"]
        out = jnp.matmul(alpha[None, :, :, :], v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, P, self.d_model)
        x = x + _dense(out, layer["W_o"])

        h2 = _layer_norm(x, layer["norm2"])
        ff = _dense(jax.nn.relu(_dense(h2, layer["ff1"])), layer["ff2"])
        return x + ff

    def forward_base(self, params, sigma):
        """
        Base forward without patch-internal translation symmetrization.
        """
        sigma = jnp.asarray(sigma, dtype=self.dtype)
        sigma = jnp.atleast_2d(sigma)
        B = sigma.shape[0]

        x = sigma.reshape(B, self.P, self.b)
        x = _dense(x, params["patch_embed"])

        idx = jnp.arange(self.P)
        rel_idx = (idx[:, None] - idx[None, :]) % self.P

        for layer in params["layers"]:
            x = self._block(x, layer, rel_idx)

        z = jnp.sum(x, axis=1)
        if self.final_layer_norm and self.d_model != 1:
            z = _layer_norm(z, params["final_norm"])
        u = _dense(z.astype(params["rbm"]["kernel"].dtype), params["rbm"])
        return jnp.sum(_complex_log_cosh(u), axis=-1)

    def _marshall_phase(self, sigma):
        """
        Marshall sign for a bipartite antiferromagnetic chain:
        psi(sigma) -> (-1) ** N_up_even psi(sigma).
        """
        sigma = jnp.asarray(sigma, dtype=self.dtype)
        sigma = jnp.atleast_2d(sigma)
        n_up_even = jnp.sum(sigma[:, ::2] > 0, axis=1)
        return jnp.pi * n_up_even.astype(self.dtype)

    def log_psi(self, params, sigma):
        """
        Return complex log wavefunction values, shape (B,).
        """
        sigma = jnp.asarray(sigma, dtype=self.dtype)
        sigma = jnp.atleast_2d(sigma)

        if (not self.symmetrize) or self.b == 1:
            log_psi = self.forward_base(params, sigma)
        else:
            shifts = jnp.arange(self.b)

            def shifted_logpsi(r):
                return self.forward_base(params, jnp.roll(sigma, shift=-r, axis=1))

            log_psis = jax.vmap(shifted_logpsi)(shifts)
            m = jnp.max(log_psis.real, axis=0, keepdims=True)
            log_psi = jnp.squeeze(m, axis=0) + jnp.log(
                jnp.sum(jnp.exp(log_psis - m), axis=0) + 1e-30
            )

        if self.marshall_sign:
            log_psi = log_psi + 1j * self._marshall_phase(sigma)
        return log_psi

    def __call__(self, params, sigma):
        """
        Return (log_amp, phase), matching the old CQP MPS model contract.
        """
        log_psi = self.log_psi(params, sigma)
        return log_psi.real.astype(self.dtype), log_psi.imag.astype(self.dtype)

    def make_apply_fn(self):
        """
        Build a jit-compiled apply function for repeated sampler/energy calls.
        """
        return jax.jit(partial(ViTWaveFunction.log_psi, self))
