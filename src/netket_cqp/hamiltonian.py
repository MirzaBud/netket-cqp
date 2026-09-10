from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

try:
    import netket as nk
except ImportError:  # pragma: no cover - optional runtime dependency
    nk = None


def _model_log_psi(model, params, sigma):
    """
    Normalize model outputs to a complex log wavefunction.
    """
    if hasattr(model, "log_psi"):
        return model.log_psi(params, sigma)

    out = model(params, sigma) if params is not None else model(sigma)
    if isinstance(out, tuple):
        return out[0] + 1j * out[1]
    return out


@dataclass
class J1J2Hamiltonian:
    """
    JAX local-energy evaluator plus NetKet operator builder for 1D J1-J2 chains.
    """

    L: int
    J1: float = 1.0
    J2: float = 0.0
    sign_rule: bool = False
    monitor_energy_imag: bool = False
    nn_pairs: jnp.ndarray = field(init=False)
    nnn_pairs: jnp.ndarray = field(init=False)
    _last_energy_stats: dict | None = field(default=None, init=False)

    def __post_init__(self):
        if self.sign_rule and self.L % 2 != 0:
            raise ValueError("Marshall sign_rule requires an even-length bipartite chain.")
        sites = jnp.arange(self.L)
        self.nn_pairs = jnp.stack([sites, (sites + 1) % self.L], axis=1)
        self.nnn_pairs = jnp.stack([sites, (sites + 2) % self.L], axis=1)

    def _update_energy_stats(self, E_total):
        """Cache local-energy statistics for external inspection/debugging."""
        E_np = np.asarray(jax.device_get(E_total))
        self._last_energy_stats = {
            "real_mean": float(E_np.real.mean()),
            "real_std": float(E_np.real.std()),
            "imag_mean": float(E_np.imag.mean()),
            "imag_std": float(E_np.imag.std()),
            "num_samples": int(E_np.size),
        }

    def get_last_energy_stats(self):
        """Return the latest cached local-energy statistics, or None."""
        return self._last_energy_stats

    def print_last_energy_stats(self, prefix: str = "LocalEnergyStats"):
        """Print latest cached local-energy stats, including imaginary-part moments."""
        if self._last_energy_stats is None:
            print(f"{prefix}: no stats cached yet (call local_energy first).")
            return
        s = self._last_energy_stats
        print(
            f"{prefix}: "
            f"Re(mean)={s['real_mean']:.6e}, Re(std)={s['real_std']:.6e}, "
            f"Im(mean)={s['imag_mean']:.6e}, Im(std)={s['imag_std']:.6e}, "
            f"N={s['num_samples']}"
        )

    def _swap_all_pairs(self, configs, pairs):
        """
        Build all pair-swapped configurations, shape (B, Nb, L).
        """
        B, L = configs.shape
        Nb = pairs.shape[0]
        flip = jnp.broadcast_to(configs[:, None, :], (B, Nb, L)).copy()
        batch_ids = jnp.arange(B)[:, None]
        bond_ids = jnp.arange(Nb)[None, :]
        i_idx = pairs[:, 0][None, :]
        j_idx = pairs[:, 1][None, :]
        si = flip[batch_ids, bond_ids, i_idx]
        sj = flip[batch_ids, bond_ids, j_idx]
        flip = flip.at[batch_ids, bond_ids, i_idx].set(sj)
        flip = flip.at[batch_ids, bond_ids, j_idx].set(si)
        return flip

    def _offdiag_from_pairs(self, model, params, configs, log_psi, pairs, J, sign=1.0):
        """
        Compute off-diagonal contribution for one neighbor set.
        """
        B, L = configs.shape
        Nb = pairs.shape[0]

        flipped = self._swap_all_pairs(configs, pairs)
        log_psi_flip = _model_log_psi(model, params, flipped.reshape(B * Nb, L))
        log_psi_flip = log_psi_flip.reshape(B, Nb)

        ratio = jnp.exp(log_psi_flip - log_psi[:, None])
        s_i = configs[:, pairs[:, 0]]
        s_j = configs[:, pairs[:, 1]]
        valid_flip_mask = (s_i != s_j).astype(ratio.dtype)
        return 0.5 * J * sign * jnp.sum(valid_flip_mask * ratio, axis=1)

    def _local_energy_impl(self, model, params, configs):
        """Pure local-energy implementation suitable for JAX transformations."""
        configs = jnp.asarray(configs)
        configs = jnp.atleast_2d(configs)

        log_psi = _model_log_psi(model, params, configs)
        s = configs.astype(jnp.float32)

        si, sj = s[:, self.nn_pairs[:, 0]], s[:, self.nn_pairs[:, 1]]
        E_diag = 0.25 * self.J1 * jnp.sum(si * sj, axis=1)

        si, sj = s[:, self.nnn_pairs[:, 0]], s[:, self.nnn_pairs[:, 1]]
        E_diag = E_diag + 0.25 * self.J2 * jnp.sum(si * sj, axis=1)

        nn_sign = -1.0 if self.sign_rule else 1.0
        E_off = self._offdiag_from_pairs(
            model, params, configs, log_psi, self.nn_pairs, self.J1, sign=nn_sign
        )
        E_off = E_off + self._offdiag_from_pairs(
            model, params, configs, log_psi, self.nnn_pairs, self.J2
        )

        complex_dtype = jnp.result_type(log_psi, E_off, jnp.complex64)
        return E_diag.astype(complex_dtype) + E_off.astype(complex_dtype)

    def local_energy(self, model, params, configs):
        """
        Compute complex local energy with complete J1/J2 off-diagonal terms.

        Args:
            model: JAX model returning complex log_psi or (log_amp, phase)
            params: parameter pytree, or None for stateful callables
            configs: (B, L) spin configs with values +/-1

        Returns:
            (B,) complex local energies
        """
        E_total = self._local_energy_impl(model, params, configs)
        self._update_energy_stats(E_total)
        if self.monitor_energy_imag:
            self.print_last_energy_stats(prefix="J1J2Hamiltonian")
        return E_total

    def make_local_energy_fn(self, model):
        """
        Build a jit-compiled local-energy function closing over a static model.
        """
        return jax.jit(lambda params, configs: self._local_energy_impl(model, params, configs))

    def mean_energy(self, model, params, configs):
        E = self.local_energy(model, params, configs)
        E_real = E.real
        return float(jnp.mean(E_real)), float(jnp.std(E_real))

    def netket_graph(self):
        """
        Build the colored J1-J2 graph used by NetKet tutorials.
        """
        if nk is None:
            raise ImportError("netket is required for netket_graph().")
        edges = []
        for i in range(self.L):
            edges.append([i, (i + 1) % self.L, 0])
            edges.append([i, (i + 2) % self.L, 1])
        return nk.graph.Graph(edges=edges)

    def netket_hilbert(self):
        """
        Build the total-Sz=0 spin Hilbert space used by MetropolisExchange.
        """
        if nk is None:
            raise ImportError("netket is required for netket_hilbert().")
        return nk.hilbert.Spin(s=0.5, total_sz=0.0, N=self.L)

    def to_netket_operator(self, sign_rule=None):
        """
        Build the NetKet JAX Heisenberg operator matching this Hamiltonian.
        """
        if nk is None:
            raise ImportError("netket is required for to_netket_operator().")

        graph = self.netket_graph()
        hilbert = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=graph.n_nodes)
        if sign_rule is None:
            sign_rule = self.sign_rule

        op = nk.operator.Heisenberg(
            hilbert=hilbert,
            graph=graph,
            J=[self.J1, self.J2],
            sign_rule=[sign_rule, False],
        )
        return op.to_jax_operator()

    def structure_factor(self):
        """
        Build NetKet antiferromagnetic structure-factor observable.
        """
        if nk is None:
            raise ImportError("netket is required for structure_factor().")

        hi = self.netket_hilbert()
        obs = nk.operator.LocalOperator(hi, dtype=complex)
        for i in range(self.L):
            for j in range(self.L):
                obs += nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, j) * ((-1) ** (i - j)) / self.L
        return obs
