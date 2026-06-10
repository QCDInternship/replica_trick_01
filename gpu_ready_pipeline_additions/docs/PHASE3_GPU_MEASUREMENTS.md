# Phase 3 GPU Measurements

## Goal

Phase 3 begins optional GPU-enabled measurement infrastructure. The first
module mirrors the existing CPU plaquette and ordinary Wilson action formulas
while allowing a full saved configuration to be copied to a GPU once at the
start of a measurement.

## Non-goals

- No configuration generation changes.
- No Markov-chain update changes.
- No CUDA kernels.
- No parallel link updates.
- No GPU measurement kernels yet.
- No production default changes.

## CPU reference

Configuration generation remains the CPU reference because it controls the
Markov chain: proposal generation, acceptance, update ordering, boundary
conditions, action definitions, tadpole-improvement behavior, and save
frequency must remain unchanged. Phase 3A only reads saved `.npy`
configurations for measurement.

CPU measurement remains the default. If `--use-gpu` is omitted, measurements run
through NumPy. If `--use-gpu` is requested but CuPy/CUDA is unavailable, the
code falls back to CPU with a warning unless `--strict` is also passed.

## Commands

Run one measurement on CPU:

```bash
python scripts/measure_configuration.py --config path/to/config.npy --beta 5.7
```

Run one measurement with the optional GPU backend:

```bash
python scripts/measure_configuration.py --config path/to/config.npy --beta 5.7 --use-gpu
```

Require GPU availability:

```bash
python scripts/measure_configuration.py --config path/to/config.npy --beta 5.7 --use-gpu --strict
```

Write a JSON summary:

```bash
python scripts/measure_configuration.py --config path/to/config.npy --beta 5.7 --output measurement.json
```

Run a CPU benchmark on a synthetic identity-link configuration:

```bash
python scripts/benchmark_gpu_measurements.py
```

Run a GPU-requested benchmark:

```bash
python scripts/benchmark_gpu_measurements.py --use-gpu
```

Run a benchmark on one saved configuration:

```bash
python scripts/benchmark_gpu_measurements.py --config path/to/config.npy --beta 5.7 --use-gpu
```

Small lattices may not show GPU speedup because host-device transfer,
synchronization, and Python/CuPy launch overhead can dominate the tiny
measurement workload.

## Fallback behavior

The measurement module imports the Phase 2 backend. CuPy remains optional and is
imported lazily. CPU-only machines continue to run normally, and GPU use is off
by default.

## Validation tests

Run the measurement validation tests with:

```bash
pytest tests/test_gpu_measurements.py
```

On CPU-only machines, CPU fallback tests run and GPU-specific CPU-vs-GPU
assertions are skipped with a clear pytest reason. On GPU machines with CuPy and
CUDA visible, the tests compare average plaquette, ordinary Wilson action, and
common numeric `measurement_summary` fields between CPU and GPU paths.

The default validation tolerances are `rtol=1e-10` and `atol=1e-12`. These are
intended for tiny deterministic arrays and should only be loosened with a
documented numerical reason.

## Phase 0 integration note

Where applicable, future GPU measurement outputs should also be checked against
the Phase 0 reference data using:

```bash
python scripts/validate_against_reference.py
```

## Active analysis scripts

The tadpole-improvement `action_v_cfg.py` script has optional `--use-gpu` and
`--strict-gpu` flags. Its CPU path remains the default, and the output `.dat`
format is unchanged.

The legacy root-level `action_v_cfg.py` and the `plaq_v_cfg.py` scripts execute
analysis at import time and do not currently have an argparse entry point. Phase
3 keeps those scripts on their existing CPU path; use
`scripts/measure_configuration.py` and `scripts/benchmark_gpu_measurements.py`
as the Phase 3 GPU measurement entry points.

## GPU optimisation checklist

- CPU and GPU measurements agree against Phase 0 references where available.
- CPU-only tests continue to pass with GPU assertions skipped.
- A saved-configuration benchmark shows enough runtime in measurement code to
  justify GPU work.
- Transfer and synchronization overhead are small compared with device compute
  time.
- The target measurement can be vectorized without changing orientation,
  normalization, boundary conditions, or action definitions.
