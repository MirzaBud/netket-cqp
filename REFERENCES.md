# NetKet references used

Source pages:

- [Vision Transformer wave function](https://netket.readthedocs.io/en/latest/tutorials/ViT-wave-function.html)
- [Ground-State: J1-J2 model](https://netket.readthedocs.io/en/latest/tutorials/gs-j1j2.html)

Implementation notes:

- The ViT tutorial uses factored multi-head attention, where attention weights
  depend on patch positions rather than the sample content. `ViTWaveFunction`
  follows that design with `rel_alpha`, per-head value projection, residual
  blocks, and a native complex RBM/log-cosh output head.
- The J1-J2 tutorial builds a colored graph containing nearest and next-nearest
  neighbor bonds, then uses a spin Hilbert space constrained to `total_sz=0`.
  `J1J2Hamiltonian.netket_graph()` and `netket_hilbert()` expose that setup.
- The NetKet optimization route is `MetropolisExchange -> MCState -> VMC_SR`.
  `VMCSampler.to_netket_sampler()` and `SROptimizer.to_netket_driver()` keep
  those entry points available, while the JAX fallback keeps the old CQP MPS
  workflow runnable without NetKet installed.
- For acceleration, the local path uses JAX arrays and batches sample-level
  work with `vmap`; sampler steps and model application can be `jit` compiled.
