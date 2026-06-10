# Phase 4 GPU Generation Preparation

## RNG stream scaffolding

Phase 4B adds optional deterministic RNG-stream helpers for future CUDA
generation work. The helpers pre-generate two explicit arrays:

- proposal indices with shape `(Nt, Nx, Ny, Nz, 4, Nhits)`
- Metropolis acceptance uniforms with shape `(Nt, Nx, Ny, Nz, 4, Nhits)`

Explicit streams make future CUDA kernels easier to reason about because the
random choices are separated from link-update execution. They also make stream
shape, dtype, range, and memory usage easy to validate before any GPU generation
kernel exists.

## Scope

This does not change the physics path yet. Production configuration generation
continues to use its existing RNG calls and sequencing. The new module is a
standalone scaffold and is not wired into Markov-chain generation.

Switching the production generator to pre-generated streams would change exact
RNG sequencing and must be validated separately against the Phase 0 references,
acceptance behavior, update ordering, boundary conditions, and saved outputs.

## Check command

Run a small stream check with:

```bash
python scripts/check_rng_streams.py
```

Custom dimensions can be supplied:

```bash
python scripts/check_rng_streams.py --nt 4 --nx 4 --ny 4 --nz 4 --nhits 10 --n-matrices 200 --seed 12345
```

Run the tests with:

```bash
pytest tests/test_rng_streams.py
```

## Readiness check

Run the combined Phase 4 readiness check with:

```bash
python scripts/check_gpu_generation_readiness.py
```

The check creates a synthetic identity-link gauge field, synthetic proposal
matrices with shape `(n_matrices, 3, 3)`, deterministic RNG streams, and a
memory estimate. It also asks the Phase 2 backend whether a GPU is available.

On a CPU-only machine, the expected report includes:

```text
GPU backend available: no
Gauge layout ready: yes
Proposal layout ready: yes
RNG streams ready: yes
CUDA generation implemented: no
```

On a GPU machine with CuPy/CUDA visible, the first line should instead report:

```text
GPU backend available: yes
```

Require GPU availability with:

```bash
python scripts/check_gpu_generation_readiness.py --strict-gpu
```

CPU-only systems pass without `--strict-gpu` and fail with `--strict-gpu`.

## Final Phase 4 checklist

- Gauge-field layout can be validated.
- Proposal matrix layout can be validated.
- Deterministic RNG streams can be generated and validated.
- GPU backend availability can be detected.
- Memory estimates are reported for gauge fields, proposals, proposal indices,
  and acceptance uniforms.
- Configuration generation is still CPU-only.
- GPU generation remains explicitly unimplemented.

## Phase 5 direction

Phase 5 should offload one non-Markov measurement/action kernel first, not full
generation. A limited measurement/action kernel is easier to validate against
the CPU reference because it does not alter Markov-chain RNG sequencing,
acceptance decisions, update ordering, or saved configurations.

## Do not claim yet

- Do not claim full CUDA generation.
- Do not claim GPU-accelerated Markov-chain sweeps.
- Safe claim: GPU-generation-ready data layout and RNG scaffolding.
