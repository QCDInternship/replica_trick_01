# Phase 7 CUDA Generation Prototype Design

## Goal

Phase 7 starts the design of a safe experimental CUDA-generation prototype.
This phase does not implement CUDA kernels and does not change the production
CPU generator. The immediate objective is to define a separate checkerboard /
red-black prototype path that can be validated before any GPU Markov-chain
generation is attempted.

## Why naive parallel link updates are unsafe

The production generator is a Markov-chain update. A proposed change to one
link can affect the local staples and action differences seen by nearby later
updates. Updating many neighboring links at once can therefore change the
transition kernel, not just the implementation speed.

Naively mapping one CUDA thread to every link and accepting proposals in
parallel would silently change the effective update ordering. It could also
change which neighboring links are visible when each staple is evaluated. That
would make the generated trajectory a different Markov chain unless the update
scheme is explicitly redesigned and validated.

RNG sequencing is also part of the trajectory. Even if the proposal matrices
and Metropolis uniforms have the same distributions, changing when each random
number is consumed changes exact trajectories and can alter reproducibility.

## Current CPU generator update ordering

The active replica generator is `markov_chain_sweep_replica` in:

```text
replica_trick_01/tadpole_improvement/gauge_latticeqcd.py
```

The current production ordering is:

```text
for config in range(Nstart, Nstart + Ncfg - 1):
  if config output does not already exist:
    for t in range(Nt):
      for x in range(Nx):
        for y in range(Ny):
          for z in range(Nz):
            for mu in range(4):
              compute replica staples for this site/link
              for hit in range(Nhits):
                draw proposal matrix index
                form U_prime = proposal @ U
                compute dS_1, dS_2, and dS_int
                accept if exp(-dS_int) > uniform random number
```

The CPU generator saves `config_*_U1.npy` after completing the full nested
loop for a configuration. Phase 7 must not change that production behavior.

## Why checkerboard/red-black is being considered

Checkerboard / red-black ordering is a possible experimental structure for
separating sets of lattice sites that are not direct nearest neighbors under a
chosen stencil. In principle, one color can be updated while the other color is
held fixed, then the second color is updated. That structure is more compatible
with batched or GPU-style execution than the current single-link lexicographic
loop.

This is only a candidate design. It changes the order in which accepted links
become visible to later updates, so it must be treated as a new experimental
Markov-chain proposal and validated statistically against the CPU reference.

## Proposed experimental checkerboard ordering

Any checkerboard prototype must live outside the production generator. The
proposed experimental ordering is:

```text
for config in range(Nstart, Nstart + Ncfg - 1):
  for color in (red, black):
    for mu in range(4):
      identify all sites with parity color
      compute local staple/action pieces using the current opposite-color field
      apply Nhits local Metropolis proposals for those candidate links
  save only under an explicit experimental output path
```

Open design questions:

- Whether color should be outermost or inside the direction loop.
- Whether `Nhits` should remain a serial loop per link inside a color batch.
- Whether proposal indices and uniforms should be pre-generated with Phase 4
  RNG stream helpers.
- How replica partition boundaries affect color independence.
- Whether the selected stencil really avoids read/write conflicts for each
  proposed batched update.

These questions must be answered in a CPU checkerboard reference first. CUDA
kernels should wait until the CPU experimental ordering has tests and reference
comparisons.

## Parts that may change trajectories

The following changes can alter exact Markov-chain trajectories:

- Reordering site visits from lexicographic order to checkerboard order.
- Updating many same-color links before neighboring opposite-color links.
- Recomputing staples at different points in the update sequence.
- Pre-generating proposal indices and Metropolis uniforms instead of consuming
  `np.random` in the production order.
- Changing floating-point grouping in local matrix products or action pieces.
- Moving local arithmetic to GPU arrays with different synchronization points.

Exact trajectory equality is therefore not the right validation target for a
full checkerboard Markov-chain prototype. Local mathematical kernels should
match strictly, while Markov-chain outputs require statistical validation.

## Observables that must be checked

Before trusting any checkerboard or CUDA-generation prototype, compare:

- plaquette histories
- ordinary and replica action histories
- acceptance rates if available
- final replica observables within uncertainties
- Phase 0 reference observables
- saved-output metadata and shapes

The CPU generator remains authoritative throughout this process.

## Phase 7 non-goals

- No production CUDA generator.
- No default GPU generation.
- No change to the CPU reference generator.
- No change to physics parameters.
- No claim of full GPU lattice-QCD simulation.

## Validation requirements

- Compare plaquette histories.
- Compare action histories.
- Compare acceptance rates if available.
- Compare final replica observables within uncertainties.
- Validate against Phase 0 reference outputs.
- Use strict tests for local mathematical kernels.
- Use statistical tests for Markov-chain trajectories.

## Prototype architecture

Planned experimental files:

```text
replica_trick_01/cuda_generation/
replica_trick_01/cuda_generation/__init__.py
replica_trick_01/cuda_generation/cpu_checkerboard_reference.py
replica_trick_01/cuda_generation/kernel_math.py
replica_trick_01/cuda_generation/cuda_kernels.py
scripts/run_cuda_generation_prototype.py
tests/test_cuda_generation_math.py
tests/test_cpu_checkerboard_reference.py
```

Current Phase 7A scope creates only the package skeleton and documentation.
`cpu_checkerboard_reference.py` is reserved for a future CPU-only experimental
checkerboard implementation. `kernel_math.py` is reserved for isolated local
math helpers that can be tested strictly against the production formulas.
`cuda_kernels.py` is reserved for future CUDA prototypes and must remain unused
by production generation.

## Phase 7B local math helpers

Phase 7B adds isolated SU(3)-style mathematical helpers in the experimental
package only:

```text
replica_trick_01/cuda_generation/kernel_math.py
```

CPU/NumPy helpers:

- `matmul3_cpu(A, B)`
- `dagger3_cpu(A)`
- `trace3_cpu(A)`
- `real_trace3_cpu(A)`
- `delta_s_cpu(U_old, U_new, staple_a, staple_b, beta, u0=1.0)`

`delta_s_cpu` mirrors the existing production mathematical expression:

```text
(-beta / 3 / u0) * real(trace((U_new - U_old) @ staple_a
                              + conj((U_new - U_old) @ staple_b).T))
```

Optional Numba-CUDA helper definitions live in:

```text
replica_trick_01/cuda_generation/cuda_kernels.py
```

These definitions are for tiny local math validation only. They are not a CUDA
generation sweep and are not called by production generation.

## Phase 7B validation

Validation uses fixed random seeds and strict tolerances:

```text
rtol = 1e-12
atol = 1e-12
```

CPU-only expected behavior:

- The package imports successfully.
- CPU helper tests run and compare against direct NumPy expressions.
- CUDA-specific tests skip cleanly.
- `python scripts/check_cuda_generation_math.py` reports CUDA unavailable and
  skips the CUDA helper check.

GPU-machine expected behavior:

- CPU helper tests still run against NumPy references.
- A tiny optional Numba-CUDA batched `3x3` matrix multiplication kernel runs.
- CUDA output is compared against NumPy with the same strict tolerances.

Smoke command:

```bash
python scripts/check_cuda_generation_math.py
```

Strict CUDA smoke command:

```bash
python scripts/check_cuda_generation_math.py --strict-cuda
```

## Phase 7C experimental prototypes

Phase 7C adds an isolated CPU checkerboard/red-black prototype and a tiny CUDA
local-operation prototype. Both are experimental and are not production
generation.

CPU checkerboard prototype:

```text
replica_trick_01/cuda_generation/cpu_checkerboard_reference.py
```

The CPU checkerboard prototype:

- operates on tiny lattices only
- validates Phase 4 gauge-field and proposal layouts
- uses explicit Phase 4 RNG streams
- uses production local replica staple/action methods where possible
- never calls `markov_chain_sweep_replica`
- updates by `sweep -> mu -> parity -> t -> x -> y -> z -> hit`
- records that exact production trajectories are not expected to match

Run the CPU checkerboard prototype:

```bash
python scripts/run_cuda_generation_prototype.py --experimental --backend cpu-checkerboard
```

Run a smaller smoke-sized CPU checkerboard prototype:

```bash
python scripts/run_cuda_generation_prototype.py --experimental --backend cpu-checkerboard --nt 2 --nx 2 --ny 1 --nz 1 --nhits 2
```

Tiny CUDA prototype:

```text
replica_trick_01/cuda_generation/cuda_kernels.py
```

The CUDA prototype is only a local batched proposal-application kernel:

```text
U_prime = M @ U
```

It is not a lattice sweep, does not evaluate staples, does not make Metropolis
accept/reject decisions, and does not save production configurations.

Run the CUDA prototype only on a CUDA-capable machine:

```bash
python scripts/run_cuda_generation_prototype.py --experimental --backend cuda-prototype --strict-cuda
```

Prototype outputs are written under:

```text
prototype_outputs/cuda_generation/
```

The runner saves:

- `prototype_config.npy`
- `metadata.json`
- `observables.json`

The observables are validation hooks computed from CPU reference measurement
routines when available. They are not evidence of physics equivalence by
themselves.

## Phase 7C validation warnings

- Checkerboard ordering changes the Markov-chain trajectory.
- Exact equality with production configurations is not expected.
- Passing tiny tests does not validate production physics.
- Local CUDA proposal application does not imply a safe CUDA generation sweep.
- Plaquette/action comparisons are hooks for investigation, not proof.
- Full validation still requires Phase 0 observable comparisons and statistical
  tests over representative ensembles.

## Phase 7C expected limitations

- Tiny lattices only.
- CPU checkerboard reference is experimental and separate.
- CUDA prototype handles only one local batched operation.
- No production output directories are written.
- No production defaults are changed.
- CPU reference generator remains authoritative.

## Do not claim yet

- Do not claim GPU-accelerated configuration generation.
- Do not claim CUDA Markov-chain sweeps.
- Do not claim full GPU lattice-QCD simulation.
- Do not claim checkerboard ordering is physics-equivalent until statistical
  validation is complete.
- Do not claim the Phase 7C CUDA local-operation prototype is a full generator.
