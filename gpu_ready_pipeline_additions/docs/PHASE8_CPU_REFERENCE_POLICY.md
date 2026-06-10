# Phase 8 CPU Reference Policy

## Policy

CPU is the authoritative reference backend. The default backend is always:

```text
cpu
```

GPU measurement and CUDA prototype paths are explicit opt-ins. They do not
replace the CPU reference generator, do not change production generation, and
do not change physics parameters.

## Registered backends

| Backend | Default | Experimental | Availability |
| --- | --- | --- | --- |
| `cpu` | yes | no | always available |
| `mpi` | no | no | available when `mpi4py` imports |
| `gpu-measure` | no | no | available when CuPy/CUDA is detected by Phase 2 backend |
| `cuda-prototype` | no | yes | experimental; requires explicit opt-in and CUDA availability |

`gpu-measure` is for measurement/action analysis only. CPU fallback remains
available in measurement scripts unless strict GPU mode is requested.

`cuda-prototype` is experimental. It must not write to production output
directories and must require an explicit `--experimental` flag in prototype
runners.

## Availability checks

List backends:

```bash
python scripts/list_backends.py
```

The registry lives in:

```text
replica_trick_01/backends.py
```

Key helpers:

- `default_backend()`
- `available_backends()`
- `is_backend_available(name)`
- `require_backend(name)`
- `backend_description(name)`
- `is_experimental_backend(name)`

## Run metadata schema

Backend-labelled outputs use:

```text
replica_trick_01/run_metadata.py
```

The metadata JSON includes:

- timestamp
- git commit hash when available
- Python and NumPy versions
- CuPy/GPU availability for GPU-relevant paths
- backend name
- experimental flag
- lattice dimensions
- beta, alpha, `Nhits`, and seed when known
- output directory
- validation status
- warning string for experimental outputs
- original parameter dictionary

Experimental metadata includes:

```text
EXPERIMENTAL OUTPUT: this is not production, is not physics-validated,
and must not be treated as a production result without validation.
```

Validation status should be treated as a gate. Outputs with
`validation_status` other than `passed` must not be presented as validated
production-equivalent results.

## Validation gates

Inspect metadata:

```bash
python scripts/check_run_metadata.py --metadata path/to/metadata.json
```

Require a validated result:

```bash
python scripts/check_run_metadata.py --metadata path/to/metadata.json --require-validated
```

Reject experimental outputs:

```bash
python scripts/check_run_metadata.py --metadata path/to/metadata.json --reject-experimental
```

Validate backend output against Phase 0 and inspect metadata:

```bash
python scripts/validate_backend_output.py \
  --candidate path/to/candidate_observables.json \
  --metadata path/to/metadata.json \
  --mode strict
```

Reject experimental candidates while validating:

```bash
python scripts/validate_backend_output.py \
  --candidate path/to/candidate_observables.json \
  --metadata path/to/metadata.json \
  --mode strict \
  --reject-experimental
```

## Example commands

CPU reference measurement:

```bash
python scripts/measure_configuration.py --config path/to/config.npy --beta 5.7
```

Explicit CPU backend measurement:

```bash
python scripts/measure_configuration.py --backend cpu --config path/to/config.npy --beta 5.7
```

CPU measurement with metadata:

```bash
python scripts/measure_configuration.py \
  --backend cpu \
  --config path/to/config.npy \
  --beta 5.7 \
  --metadata-output runs/cpu_measurement_metadata.json
```

GPU measurement request:

```bash
python scripts/measure_configuration.py --backend gpu-measure --kernel backend --config path/to/config.npy --beta 5.7
```

CUDA prototype run:

```bash
python scripts/run_cuda_generation_prototype.py --experimental --backend cuda-prototype --strict-cuda
```

## Final Phase 8 checklist

- CPU default confirmed.
- Backend registry exists.
- Metadata labelling exists.
- Experimental outputs labelled.
- Validation gate exists.
- Smoke checks exist.
- GPU is optional.
- CUDA generation is not production.

## Safe claim after Phase 8

The repository preserves a CPU reference implementation as the authoritative
physics path, with optional GPU measurement support and clearly labelled
experimental CUDA-generation prototypes guarded by validation and metadata
checks.

## Do not claim yet

- Do not claim production GPU configuration generation.
- Do not claim full CUDA Markov-chain sweeps.
- Do not claim full GPU lattice-QCD simulation.
- Do not remove or bypass CPU reference validation.
