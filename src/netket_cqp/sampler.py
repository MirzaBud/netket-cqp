from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class VMCSampler:
    """
    JAX Metropolis sampler with M parallel chains.
    Proposal uses global up-down swaps in each chain to preserve total Sz.
    """

    model: object
    L: int
    n_chains: int = 4
    move_distances: tuple = (1, 2)
    seed: int | None = None
    print_every: int = 0
    verbose: bool = False

    def __post_init__(self):
        self.key = jax.random.PRNGKey(0 if self.seed is None else self.seed)
        self.current_state = None
        self.current_log_psi = None
        self._step_count = 0
        self._step_fn = jax.jit(self._step_impl)

    def reset(self, params=None, initial_states=None):
        """
        Initialize each chain to an Sz_tot=0 configuration.
        """
        if initial_states is not None:
            states = jnp.asarray(initial_states, dtype=jnp.float32)
        else:
            base = jnp.concatenate(
                [
                    jnp.ones((self.L // 2,), dtype=jnp.float32),
                    -jnp.ones((self.L - self.L // 2,), dtype=jnp.float32),
                ]
            )

            def shuffled(k):
                return jax.random.permutation(k, base)

            self.key, subkey = jax.random.split(self.key)
            keys = jax.random.split(subkey, self.n_chains)
            states = jax.vmap(shuffled)(keys)

        self.current_state = states
        self.refresh(params)

    def refresh(self, params=None):
        """
        Recompute cached log amplitudes/phases for current_state under current parameters.
        """
        if self.current_state is None:
            raise RuntimeError("current_state is None; call reset() first.")
        self.current_log_psi = _model_log_psi(self.model, params, self.current_state)

    def _step_impl(self, key, params, state, log_psi):
        M = state.shape[0]
        key_i, key_j, key_u = jax.random.split(key, 3)

        up_scores = jnp.where(state > 0, jax.random.uniform(key_i, state.shape), -1.0)
        down_scores = jnp.where(state < 0, jax.random.uniform(key_j, state.shape), -1.0)
        i_idx = jnp.argmax(up_scores, axis=1)
        j_idx = jnp.argmax(down_scores, axis=1)

        rows = jnp.arange(M)
        new_state = state
        si = state[rows, i_idx]
        sj = state[rows, j_idx]
        new_state = new_state.at[rows, i_idx].set(sj)
        new_state = new_state.at[rows, j_idx].set(si)

        log_new = _model_log_psi(self.model, params, new_state)
        delta = 2.0 * (log_new.real - log_psi.real)
        u = jnp.log(jax.random.uniform(key_u, (M,)))
        accept = (delta >= 0.0) | (u < delta)

        state = jnp.where(accept[:, None], new_state, state)
        log_psi = jnp.where(accept, log_new, log_psi)
        return state, log_psi, accept

    def step(self, params=None):
        """
        One Metropolis proposal per chain.
        """
        if self.current_state is None:
            raise RuntimeError("current_state is None; call reset() first.")

        self.key, subkey = jax.random.split(self.key)
        self.current_state, self.current_log_psi, accept = self._step_fn(
            subkey, params, self.current_state, self.current_log_psi
        )

        self._step_count += 1
        if self.print_every > 0 and (self._step_count % self.print_every == 0):
            accept_rate = float(jnp.mean(accept.astype(jnp.float32)))
            print(f"[step {self._step_count}] Acceptance rate: {accept_rate*100:.2f}%")

        return accept

    def burn_in(self, params=None, n_steps: int = 200):
        """
        Run n_steps Metropolis moves without recording.
        """
        for _ in range(n_steps):
            self.step(params)

    def sample(self, params=None, n_samples=100, burn_in=200, thin=10):
        """
        Generate samples after burn-in with thinning.
        """
        if self.current_state is None:
            self.reset(params)

        if self.verbose:
            print(f"Starting {burn_in} burn-in steps...")
        self.burn_in(params, burn_in)

        kept = 0
        states_buf = []
        amps_buf = []
        phases_buf = []

        if self.verbose:
            print(f"Sampling {n_samples} states with thinning={thin}")
        steps = 0

        while kept < n_samples:
            self.step(params)
            steps += 1

            if steps % thin == 0:
                states_buf.append(self.current_state)
                amps_buf.append(self.current_log_psi.real)
                phases_buf.append(self.current_log_psi.imag)
                kept += 1

        states = jnp.stack(states_buf, axis=0)
        logamps = jnp.stack(amps_buf, axis=0)
        phases = jnp.stack(phases_buf, axis=0)
        return states, logamps, phases

    @staticmethod
    def to_netket_sampler(hilbert, graph, n_chains=4096, d_max=2, sweep_size=None):
        """
        Build the NetKet MetropolisExchange sampler used in the tutorials.
        """
        if nk is None:
            raise ImportError("netket is required for to_netket_sampler().")
        return nk.sampler.MetropolisExchange(
            hilbert=hilbert,
            graph=graph,
            d_max=d_max,
            n_chains=n_chains,
            sweep_size=sweep_size,
        )

