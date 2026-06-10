# Phase 1 CPU/HPC Baseline

Phase 1 keeps the CPU generator as the physics reference while reducing only
avoidable overhead and documenting the validation path. It must preserve the
Markov-chain transition rule, proposal set, update ordering, action/staple
definitions, replica boundary conditions, tadpole-improvement logic, and
configuration save frequency.

## Active Files Touched By Phase 1

- `gauge_latticeqcd.py`
- `Modified_Replica_generate_multiprocessing`
- `replica_trick_01/tadpole_improvement/gauge_latticeqcd.py`
- `replica_trick_01/tadpole_improvement/Modified_Replica_generate_multiprocessing.py`
- `replica_trick_01/tadpole_improvement/Replica_generate_multiprocessing.py`
- `replica_trick_01/tadpole_improvement/replica_generate.py`
- `replica_trick_01/tadpole_improvement/tadpole.py`
- `scripts/profile_replica_generation.py`
- `scripts/make_physics_reference.py`
- `scripts/check_phase1_proposals.py`
- `docs/PHASE1_CPU_HPC_BASELINE.md`

## Excluded Files And Directories

- `archive/`
- Any file under `archive/`
- Production output directories such as `RunReplica*/`
- Generated reference `.npy` files under `reference_outputs/`
- Generated profiling output under `profiles/`

## Phase 1 Non-Goals

- No GPU, CUDA, CuPy, or GPU-specific backend code.
- No global replacement of NumPy.
- No change to `Nhits`, `epsilon`, beta, alpha list, lattice dimensions, or save frequency.
- No change to proposal matrix construction or proposal ordering.
- No change to the Metropolis acceptance rule.
- No change to Markov-chain update ordering.
- No change to replica boundary conditions.
- No change to staple definitions, action definitions, or tadpole-improvement formulae.
- No partition-boundary caching.
- No parallelisation of link updates inside one lattice sweep.
- No long production simulations as part of this phase.

## Completed Changes

- Confirmed Phase 0 reference files exist:
  `docs/PHYSICS_REFERENCE_PLAN.md`,
  `scripts/make_physics_reference.py`, and
  `scripts/validate_against_reference.py`.
- Created branch `hpc-cpu-baseline`.
- Confirmed active replica generation sweeps keep only the outer
  configuration-level `tqdm`; no inner per-time-slice `tqdm(..., leave=False)`
  wrappers remain in the inspected active sweep loops.
- Converted proposal matrix containers after generation to contiguous
  `np.complex128` arrays with
  `np.ascontiguousarray(np.asarray(..., dtype=np.complex128))`.
- Added `scripts/check_phase1_proposals.py` to verify proposal array shape,
  dtype, C-contiguity, finiteness, and first Hermitian-conjugate partner order.
- Replaced per-job process fan-out in active modified generation drivers with
  bounded `multiprocessing.Pool` execution over independent `(alpha, ensemble)`
  jobs.
- Added `--workers` and `--seed` options to the active modified generation
  drivers. If `--workers` is omitted, the worker count defaults to
  `min(os.cpu_count() or 1, len(jobs), 4)`.
- Added deterministic per-job seeding with
  `base_seed + 10000 * alpha_index + ensemble_index`.
- Added optional MPI driver
  `replica_trick_01/tadpole_improvement/Modified_Replica_generate_mpi.py`.
  It broadcasts the output base directory, `u0`, job list, and base seed from
  rank 0, then distributes only independent jobs by rank striding.
- Added `scripts/profile_replica_generation.py`, a tiny cProfile/pstats
  generation profiler that writes a raw profile, top-functions report, and JSON
  metadata without touching production outputs.
- Reused `replica_trick_01/tadpole_improvement/numba_compat.py`, which provides
  a no-op `njit` fallback when Numba is unavailable.
- Added small Numba-compatible helper functions in
  `replica_trick_01/tadpole_improvement/numba_helpers.py` for 3x3 matrix
  multiplication, real trace, `deltaS` math, and `delta_S_int` math.
- Added `tests/test_numba_helpers.py` to compare helper output against direct
  NumPy/Python expressions with `rtol=1e-12` and `atol=1e-12`, and to verify
  the compatibility wrapper fallback behavior.

The Numba helpers are deliberately limited to pure array/scalar functions. The
full `ReplicaLattice` class, sweep loop, staple construction, boundary logic,
proposal generation, acceptance rule, update ordering, save cadence, and
partition-boundary handling were not refactored in Phase 1C.
The tadpole-improvement `ReplicaLattice.deltaS` and `ReplicaLattice.delta_S_int`
methods use the equivalent helper functions; `matmul3` and `real_trace` remain
validated preparation helpers and are not wired into production generation.

## Validation Command

After creating a Phase 0 reference and a candidate observables JSON, validate
implementation-only rewrites with:

```sh
python scripts/validate_against_reference.py --reference reference_outputs/phase0_tiny/reference_observables.json --candidate reference_outputs/new_run/reference_observables.json --mode strict
```

This phase must still be validated against Phase 0 outputs before being treated
as a physics-preserving baseline.

Multiprocessing and MPI changes affect job scheduling and RNG streams. Future
validation should use Phase 0 observables and statistical checks rather than
assuming bitwise-identical configurations unless the same RNG path is explicitly
preserved.

## Generation Commands

Bounded multiprocessing:

```sh
python replica_trick_01/tadpole_improvement/Modified_Replica_generate_multiprocessing.py --workers 2 --seed 12345
```

Optional MPI:

```sh
mpirun -np 4 python replica_trick_01/tadpole_improvement/Modified_Replica_generate_mpi.py --seed 12345
```

## Benchmark And Profiling Commands

Use the existing tiny profiler for local timing checks:

```sh
python scripts/profile_replica_generation.py --output-dir profiles/tmp_generation --profile-out profiles/replica_generation.prof --report-out profiles/replica_generation_top.txt --nt 4 --nx 4 --ny 4 --nz 4 --n-cfg 3 --n-hits 10 --alpha 0.0 --seed 12345
```

Expected top functions for the current CPU path include
`markov_chain_sweep_replica`, `dS_staple_replica`, `deltaS`, small 3x3 matrix
multiplication calls, proposal generation (`create_su3_set`, `matrix_su3`,
`matrix_su2`), and save-related NumPy I/O when output is enabled.

Additional benchmark commands to add later:

- A fixed tiny reference-generation timing command.
- A medium CPU-only generation timing command that does not overwrite production outputs.
- A comparison command that records wall time, acceptance output if already printed, and top profiler functions.

## Commands Run

- `test -f docs/PHYSICS_REFERENCE_PLAN.md && test -f scripts/make_physics_reference.py && test -f scripts/validate_against_reference.py && printf 'phase0 files present\n'`
- `git switch -c hpc-cpu-baseline`
- `rg -n "tqdm\\(range\\(|create_su3_set\\(|asarray\\(|ascontiguousarray" ...`
- `python -m py_compile gauge_latticeqcd.py`
- `python -m py_compile scripts/check_phase1_proposals.py`
- `python -m py_compile replica_trick_01/tadpole_improvement/gauge_latticeqcd.py replica_trick_01/tadpole_improvement/Modified_Replica_generate_multiprocessing.py replica_trick_01/tadpole_improvement/Replica_generate_multiprocessing.py replica_trick_01/tadpole_improvement/replica_generate.py replica_trick_01/tadpole_improvement/tadpole.py`
- `python -m py_compile Modified_Replica_generate_multiprocessing scripts/profile_replica_generation.py scripts/make_physics_reference.py`
- `python scripts/check_phase1_proposals.py --n-matrix 8`
- `rg -n "for t in tqdm\\(range\\([^\\n]*leave=False|leave=False\\)" gauge_latticeqcd.py replica_trick_01/tadpole_improvement/gauge_latticeqcd.py -g '!archive/**'`
- `python -m py_compile Modified_Replica_generate_multiprocessing`
- `python -m py_compile replica_trick_01/tadpole_improvement/Modified_Replica_generate_multiprocessing.py replica_trick_01/tadpole_improvement/Modified_Replica_generate_mpi.py`
- `python Modified_Replica_generate_multiprocessing --help`
- `python replica_trick_01/tadpole_improvement/Modified_Replica_generate_multiprocessing.py --help`
- `python replica_trick_01/tadpole_improvement/Modified_Replica_generate_mpi.py --seed 12345`
- `rg -n "\\bProcess\\b|from multiprocessing import Process|\\.start\\(\\)|build_generation_jobs|seed_for_job|Selected worker count|Tadpole improvement enabled" Modified_Replica_generate_multiprocessing replica_trick_01/tadpole_improvement/Modified_Replica_generate_multiprocessing.py replica_trick_01/tadpole_improvement/Modified_Replica_generate_mpi.py`
- `python -m py_compile scripts/profile_replica_generation.py replica_trick_01/tadpole_improvement/numba_helpers.py tests/test_numba_helpers.py`
- `pytest tests/test_numba_helpers.py`
