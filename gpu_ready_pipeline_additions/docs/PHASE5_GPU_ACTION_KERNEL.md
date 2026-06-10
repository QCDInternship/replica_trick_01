# Phase 5 GPU Action Kernel

## Chosen kernel

The Phase 5 target is the ordinary plaquette / ordinary Wilson-action
measurement on a saved gauge configuration. This is a non-Markov measurement
kernel and is safer to isolate before any configuration-generation work.

## CPU reference functions

The CPU reference is the active implementation in:

```text
replica_trick_01/tadpole_improvement/gauge_latticeqcd.py
```

Reference functions:

- `fn_average_plaquette(U)`
- `fn_eval_point_S(U, t, x, y, z, beta, u0=1.0)`

The Phase 5 wrapper in `replica_trick_01/action_kernel_reference.py` calls
these functions directly. It sums `fn_eval_point_S` over all lattice sites for
the ordinary Wilson action and does not modify the physics implementation.

Replica-action routines such as `calc_action_replica` were audited but are not
the Phase 5A target.

## Shape convention

Gauge fields use shape:

```text
(Nt, Nx, Ny, Nz, 4, 3, 3)
```

Axes:

- axis 0: `Nt`
- axis 1: `Nx`
- axis 2: `Ny`
- axis 3: `Nz`
- axis 4: direction `mu`
- axes 5 and 6: SU(3) matrix rows and columns

## Normalisation convention

`fn_average_plaquette(U)` sums plaquettes with `mu in range(1, 4)` and
`nu in range(mu)`, then returns:

```text
ReTr(sum plaquettes) / 3 / Nt / Nx / Ny / Nz / 6
```

`fn_eval_point_S` uses:

```text
beta * sum_mu>nu (1 - ReTr(P_mu_nu) / 3 / u0**4)
```

The ordinary Wilson action wrapper sums this point action over all lattice
sites.

## Why this is safe before generation

The kernel reads one saved gauge field and produces scalar measurements. It does
not affect proposal generation, RNG sequencing, Metropolis acceptance, update
ordering, boundary conditions, tadpole updates, or save frequency. CPU remains
the authoritative physics reference.

## Non-goals

- No GPU configuration generation.
- No CUDA kernels.
- No changes to `markov_chain_sweep_replica`.
- No parallel link updates.
- No changes to replica-action formulas.
- No changes to production defaults.

## Vectorized backend design

Phase 5B adds `replica_trick_01/action_kernel_backend.py`, an optional
vectorized backend for the chosen measurement kernel. It supports NumPy on CPU
and CuPy on GPU through the Phase 2 backend. GPU use is off by default.

The backend moves the full gauge field to the selected array module once, loops
only over the six plaquette orientations, and uses vectorized array operations
over all lattice sites. It computes the same ordered plaquette product as the
CPU reference:

```text
U_mu(x) U_nu(x + mu) U_mu^dagger(x + nu) U_nu^dagger(x)
```

Only scalar results are copied back to CPU. The backend is for measurement and
analysis only; it is not used by configuration generation.

## Check command

Run the CPU reference wrapper check with:

```bash
python scripts/check_action_kernel_reference.py
```

Run it on one saved configuration with:

```bash
python scripts/check_action_kernel_reference.py --config path/to/config.npy --beta 5.7 --u0 1.0
```

Run tests with:

```bash
pytest tests/test_action_kernel_reference.py
```

## Backend commands

Run the backend on CPU:

```bash
python scripts/run_action_kernel_backend.py --compare-reference
```

Request GPU execution:

```bash
python scripts/run_action_kernel_backend.py --use-gpu --compare-reference
```

Run on a saved configuration:

```bash
python scripts/run_action_kernel_backend.py --config path/to/config.npy --beta 5.7 --u0 1.0 --compare-reference
```

Run validation tests:

```bash
pytest tests/test_action_kernel_reference.py tests/test_action_kernel_backend.py
```

Small lattices may not show GPU speedup because transfer, synchronization, and
array-library launch overhead can dominate the tiny measurement workload.

## Benchmark commands

Benchmark the CPU reference and vectorized CPU backend on a synthetic identity
configuration:

```bash
python scripts/benchmark_action_kernel_backend.py
```

Request GPU benchmarking:

```bash
python scripts/benchmark_action_kernel_backend.py --use-gpu
```

Benchmark one saved configuration:

```bash
python scripts/benchmark_action_kernel_backend.py --config path/to/config.npy --beta 5.7 --u0 1.0 --use-gpu
```

On CPU-only machines, GPU timing is skipped unless `--use-gpu` is passed; with
`--strict`, the command exits nonzero if CuPy/CUDA is unavailable. On GPU
machines, the report includes backend GPU timing and numerical agreement
against the CPU reference.

Speedup should be interpreted as a measurement-kernel benchmark only. Small
lattices may be slower on GPU because data transfer, synchronization, and
array-library overhead can dominate.

Validation uses `rtol=1e-10` and `atol=1e-12` for average plaquette and ordinary
Wilson action comparisons.

## Phase 3 integration

`replica_trick_01/gpu_measurements.py`, `scripts/measure_configuration.py`, and
`scripts/benchmark_gpu_measurements.py` now accept an explicit kernel route:

```bash
python scripts/measure_configuration.py --config path/to/config.npy --beta 5.7 --kernel reference
python scripts/measure_configuration.py --config path/to/config.npy --beta 5.7 --kernel backend --use-gpu
```

The default remains `--kernel reference`, preserving the earlier Phase 3 path.

## Current limitations

- Only the ordinary plaquette/action measurement kernel is implemented.
- Replica-action GPU acceleration is not implemented.
- Configuration generation remains CPU-only.
- Performance depends on lattice size and transfer overhead.

## Safe claim after Phase 5

Added an optional CPU/GPU backend for a non-Markov plaquette/action measurement
kernel, validated against the CPU reference.

## Do not claim yet

- Do not claim GPU-accelerated configuration generation.
- Do not claim CUDA Markov-chain sweeps.
- Do not claim full replica-action GPU acceleration unless specifically
  implemented and validated.

## Phase 6 decision note

Profile again after Phase 5. Decide whether configuration generation is still
the dominant bottleneck. Only proceed toward CUDA generation if profiling
justifies it and preserves the CPU reference behavior.
