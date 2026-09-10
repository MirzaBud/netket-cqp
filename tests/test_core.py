import jax
import jax.numpy as jnp
import pytest

from netket_cqp import J1J2Hamiltonian, ViTWaveFunction


def test_vit_returns_one_complex_log_amplitude_per_configuration():
    model = ViTWaveFunction(L=4, b=2, d_model=4, n_heads=2, d_ff=8)
    params = model.init(jax.random.PRNGKey(0))
    configs = jnp.array([[1, -1, 1, -1], [-1, 1, -1, 1]], dtype=jnp.float32)

    log_psi = model.log_psi(params, configs)

    assert log_psi.shape == (2,)
    assert jnp.issubdtype(log_psi.dtype, jnp.complexfloating)
    assert bool(jnp.all(jnp.isfinite(log_psi)))


def test_j1j2_local_energy_for_constant_wavefunction():
    def constant_model(_params, configs):
        return jnp.zeros(configs.shape[0], dtype=jnp.complex64)

    hamiltonian = J1J2Hamiltonian(L=4, J1=1.0, J2=0.0)
    config = jnp.array([[1, -1, 1, -1]], dtype=jnp.float32)

    energy = hamiltonian.local_energy(constant_model, None, config)

    # Four anti-aligned nearest-neighbour bonds: diagonal -1, off diagonal +2.
    assert float(energy.real[0]) == pytest.approx(1.0)
    assert float(energy.imag[0]) == pytest.approx(0.0)


def test_jitted_local_energy_matches_eager_calculation():
    def constant_model(_params, configs):
        return jnp.zeros(configs.shape[0], dtype=jnp.complex64)

    hamiltonian = J1J2Hamiltonian(L=4)
    config = jnp.array([[1, -1, 1, -1]], dtype=jnp.float32)

    assert jnp.allclose(
        hamiltonian.make_local_energy_fn(constant_model)(None, config),
        hamiltonian.local_energy(constant_model, None, config),
    )
