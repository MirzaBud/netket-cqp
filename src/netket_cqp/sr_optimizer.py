from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

try:
    import netket as nk
except ImportError:  # pragma: no cover - optional runtime dependency
    nk = None


def _model_log_psi(model, params, sigma):
    if hasattr(model, "log_psi"):
        return model.log_psi(params, sigma)
    out = model(params, sigma) if params is not None else model(sigma)
    if isinstance(out, tuple):
        return out[0] + 1j * out[1]
    return out


def _ravel_real_pytree(params):
    """
    Flatten real params directly and complex params as independent real/imag parts.
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    pieces = []
    metadata = []

    for leaf in leaves:
        arr = jnp.asarray(leaf)
        size = arr.size
        is_complex = jnp.issubdtype(arr.dtype, jnp.complexfloating)
        metadata.append((is_complex, arr.shape, arr.dtype, size))
        if is_complex:
            pieces.append(arr.real.reshape(-1))
            pieces.append(arr.imag.reshape(-1))
        else:
            pieces.append(arr.reshape(-1))

    if pieces:
        flat = jnp.concatenate(pieces, axis=0)
    else:
        flat = jnp.asarray([], dtype=jnp.float32)

    def unravel(flat_params):
        out = []
        idx = 0
        for is_complex, shape, dtype, size in metadata:
            if is_complex:
                real = flat_params[idx : idx + size].reshape(shape)
                idx += size
                imag = flat_params[idx : idx + size].reshape(shape)
                idx += size
                out.append((real + 1j * imag).astype(dtype))
            else:
                out.append(flat_params[idx : idx + size].reshape(shape).astype(dtype))
                idx += size
        return jax.tree_util.tree_unflatten(treedef, out)

    return flat, unravel


@dataclass
class SROptimizer:
    """
    Stochastic Reconfiguration (SR) optimizer for complex JAX wavefunctions.

    This mirrors the old PyTorch optimizer, but computes per-sample Jacobians
    with jax.vmap/jax.grad and keeps parameters as a functional pytree.
    """

    model: object
    lr: float = 0.01
    diag_shift: float = 1e-4
    cg_maxiter: int = 10
    cg_tol: float = 1e-5
    batch_size: int | None = 256
    seed: int = 0
    _last_step_stats: dict | None = field(default=None, init=False)

    def __post_init__(self):
        self.key = jax.random.PRNGKey(self.seed)
        self._compute_O_jit = jax.jit(self._compute_O)

    def _log_psi_scalar(self, params, config):
        return _model_log_psi(self.model, params, config[None, :])[0]

    def _sample_grad(self, flat_params, unravel, config):
        def log_psi_flat(flat):
            return self._log_psi_scalar(unravel(flat), config)

        O_real = jax.grad(lambda flat: log_psi_flat(flat).real)(flat_params)
        O_imag = jax.grad(lambda flat: log_psi_flat(flat).imag)(flat_params)
        return O_real.astype(flat_params.dtype), O_imag.astype(flat_params.dtype)

    def _compute_O(self, params, configs):
        """
        Compute O matrix for each sample:
          O = d(log|psi|)/dw + i * d(phase)/dw
        """
        flat_params, unravel = _ravel_real_pytree(params)
        return jax.vmap(lambda cfg: self._sample_grad(flat_params, unravel, cfg))(configs)

    def _subsample_batch(self, configs, local_energies):
        """Use at most `batch_size` samples in one SR step to control cost."""
        if self.batch_size is None:
            return configs, local_energies
        if self.batch_size <= 0 or configs.shape[0] <= self.batch_size:
            return configs, local_energies

        self.key, subkey = jax.random.split(self.key)
        idx = jax.random.permutation(subkey, configs.shape[0])[: self.batch_size]
        return configs[idx], local_energies[idx]

    def _compute_force(self, O_real, O_imag, E_loc):
        """
        f = -2 * Re( <(E - <E>) * conj(O - <O>)> )
        """
        E_real = E_loc.real.astype(O_real.dtype)
        E_imag = E_loc.imag.astype(O_real.dtype)

        O_r_c = O_real - jnp.mean(O_real, axis=0, keepdims=True)
        O_i_c = O_imag - jnp.mean(O_imag, axis=0, keepdims=True)
        E_r_c = E_real - jnp.mean(E_real)
        E_i_c = E_imag - jnp.mean(E_imag)

        f = -2.0 * jnp.mean(O_r_c * E_r_c[:, None] + O_i_c * E_i_c[:, None], axis=0)
        return f, O_r_c, O_i_c

    def _matvec(self, O_r_c, O_i_c, v):
        """
        Apply (S + diag_shift I) to vector v without building S explicitly.
        """
        M = O_r_c.shape[0]
        x = O_r_c @ v
        y = O_i_c @ v
        Sv = (O_r_c.T @ x + O_i_c.T @ y) / M
        return Sv + self.diag_shift * v

    def _conjugate_gradient(self, O_r_c, O_i_c, f):
        x = jnp.zeros_like(f)
        r = f
        p = r
        rr_old = jnp.dot(r, r)

        for _ in range(self.cg_maxiter):
            Ap = self._matvec(O_r_c, O_i_c, p)
            alpha = rr_old / (jnp.dot(p, Ap) + 1e-16)
            x = x + alpha * p
            r = r - alpha * Ap

            rr_new = jnp.dot(r, r)
            if float(jnp.sqrt(rr_new)) < self.cg_tol:
                break

            beta = rr_new / (rr_old + 1e-16)
            p = r + beta * p
            rr_old = rr_new

        return x

    def step(self, params, configs, local_energies):
        """
        Perform one SR update using complex local energies.

        Args:
            params: parameter pytree
            configs: (M, L) or (n_samples, n_chains, L)
            local_energies: (M,) real or complex
        """
        configs = jnp.asarray(configs)
        local_energies = jnp.asarray(local_energies).reshape(-1)

        if configs.ndim == 3:
            configs = configs.reshape((-1, configs.shape[-1]))
        elif configs.ndim != 2:
            raise ValueError(f"configs must be 2D or 3D, got shape={configs.shape}")

        if configs.shape[0] != local_energies.shape[0]:
            raise ValueError(
                f"configs/local_energies size mismatch: "
                f"configs M={configs.shape[0]} vs energies={local_energies.shape[0]}"
            )

        if not bool(jnp.all(jnp.isfinite(local_energies))):
            raise FloatingPointError(
                "local_energies contains non-finite values. "
                "For this complex ansatz, a common cause is destructive interference from "
                "symmetrize=True making psi(s) numerically tiny for sampled configurations."
            )

        cfg_batch, e_batch = self._subsample_batch(configs, local_energies)
        O_real, O_imag = self._compute_O_jit(params, cfg_batch)
        f, O_r_c, O_i_c = self._compute_force(O_real, O_imag, e_batch)
        delta = self._conjugate_gradient(O_r_c, O_i_c, f)

        flat, unravel = _ravel_real_pytree(params)
        new_params = unravel(flat + self.lr * delta.astype(flat.dtype))

        f_norm = float(jnp.linalg.norm(f))
        delta_norm = float(jnp.linalg.norm(delta))
        self._last_step_stats = {
            "batch_size_used": int(cfg_batch.shape[0]),
            "force_norm": f_norm,
            "delta_norm": delta_norm,
            "force_max_abs": float(jnp.max(jnp.abs(f))),
            "delta_max_abs": float(jnp.max(jnp.abs(delta))),
        }

        if any(bool(jnp.any(~jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(new_params)):
            raise FloatingPointError(
                "SR produced non-finite parameters. Try disabling symmetrize or increasing diag_shift."
            )
        return new_params, delta_norm

    def get_last_step_stats(self):
        """Return diagnostics from the most recent SR step."""
        return self._last_step_stats

    @staticmethod
    def to_netket_driver(hamiltonian, optimizer, variational_state, diag_shift=1e-4, mode="complex"):
        """
        Build NetKet's VMC_SR driver when netket/flax are installed.
        """
        if nk is None:
            raise ImportError("netket is required for to_netket_driver().")
        return nk.driver.VMC_SR(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            variational_state=variational_state,
            diag_shift=diag_shift,
            mode=mode,
        )
