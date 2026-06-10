# Phase 9 Public GPU Benchmarking

## Goal

Phase 9 makes the repository easy to test on public or rented NVIDIA GPU
machines while preserving the Phase 8 policy: CPU reference is authoritative,
GPU measurement is optional, and CUDA generation prototypes are experimental.

This phase adds environment capture and a benchmark plan. It does not run long
benchmarks and does not implement production GPU configuration generation.

## Phase 9 non-goals

- No production CUDA generator.
- No changes to production configuration generation.
- No changes to `markov_chain_sweep_replica`.
- No changes to physics parameters, proposal generation, acceptance rules,
  update ordering, replica boundary conditions, action definitions,
  tadpole-improvement logic, RNG behavior, or save frequency.
- No mandatory CuPy dependency for CPU-only users.
- No new CUDA-generation kernels.
- No claim of production GPU configuration generation.

## Public GPU targets

Phase 9 should support environment capture and smoke benchmarking on:

- Google Colab
- Kaggle Notebooks
- generic Linux NVIDIA GPU machines
- rented GPU machines such as RunPod/Vast-style environments

Environment detection is best-effort and uses safe signals such as environment
variables and `nvidia-smi` availability.

## Benchmark categories

Public GPU runs should collect:

- GPU backend detection
- GPU measurement/action kernel benchmark
- generation-readiness checks
- optional experimental CUDA-prototype smoke test
- validation against Phase 0 reference where applicable

GPU generation remains experimental. Benchmarks should keep production CPU
reference measurements available for comparison.

## Environment capture

Capture public GPU environment metadata:

```bash
python scripts/capture_public_gpu_environment.py
```

Require a visible CUDA device:

```bash
python scripts/capture_public_gpu_environment.py --strict-gpu
```

The default output is:

```text
benchmarks/public_gpu_environment.json
```

The JSON records timestamp, git commit, Python/platform/NumPy metadata, CuPy
availability, CUDA device availability, GPU name, GPU memory info,
`nvidia-smi` output when available, Phase 8 backend registry status, and a
best-effort public-runtime classification.

## Public GPU benchmark runner

Run the orchestrated public-GPU benchmark suite in CPU-fallback mode:

```bash
python scripts/run_public_gpu_benchmarks.py --repeat 1 --warmup 0
```

Require a visible GPU:

```bash
python scripts/run_public_gpu_benchmarks.py --strict-gpu
```

Include tiny experimental prototype smoke checks:

```bash
python scripts/run_public_gpu_benchmarks.py --strict-gpu --include-experimental
```

The runner writes:

```text
environment.json
backend_check.json
gpu_measurement_benchmark.json
action_kernel_benchmark.json
generation_readiness.json
public_gpu_summary.json
public_gpu_summary.md
```

By default, the runner does not require a GPU. If no GPU is available, it runs
CPU fallback benchmarks and marks GPU timings as skipped. Experimental CUDA
prototype checks run only when `--include-experimental` is passed and still use
the existing `--experimental` guard in the prototype runner.

## Useful commands

Check optional GPU backend:

```bash
python scripts/check_gpu_backend.py
```

List Phase 8 backends:

```bash
python scripts/list_backends.py
```

Run CPU-only smoke checks:

```bash
python scripts/run_smoke_checks.py
```

Request GPU smoke checks:

```bash
python scripts/run_smoke_checks.py --include-gpu
```

Run a GPU measurement/action benchmark:

```bash
python scripts/benchmark_action_kernel_backend.py --use-gpu
```

Run the public-GPU benchmark orchestrator:

```bash
python scripts/run_public_gpu_benchmarks.py --repeat 1 --warmup 0
```

Run generation-readiness checks:

```bash
python scripts/check_gpu_generation_readiness.py
```

Run the experimental CUDA prototype smoke only with explicit opt-in:

```bash
python scripts/run_smoke_checks.py --include-experimental
```

## Optional GPU dependencies

Use `requirements-gpu.txt` only on GPU machines. Choose the CuPy wheel that
matches the CUDA runtime, and do not install multiple CuPy CUDA wheels at the
same time. CPU-only development does not require this file.

Example setup commands:

```bash
python -m pip install -r requirements-gpu.txt
python scripts/capture_public_gpu_environment.py
python scripts/check_gpu_backend.py
python scripts/list_backends.py
```

## Safe claims after Phase 9

- The repository can capture public-GPU environment metadata.
- The repository can report whether optional GPU measurement support is
  available.
- Benchmark outputs can be labelled with backend and environment metadata.
- CPU reference remains authoritative.
- CUDA-generation prototypes remain experimental.

## Claims not allowed after Phase 9

- Do not claim production GPU configuration generation.
- Do not claim full CUDA Markov-chain sweeps.
- Do not claim full GPU lattice-QCD simulation.
- Do not claim public-GPU benchmark results validate changed Markov-chain
  trajectories without Phase 0 and statistical validation.

## Public GPU runbook and report generation

Use the public GPU runbook for Colab, Kaggle, generic Linux NVIDIA machines,
and rented GPU environments:

```text
docs/PUBLIC_GPU_RUNBOOK.md
```

Generate a comparison report from one or more public benchmark runs:

```bash
python scripts/make_public_gpu_report.py --runs benchmarks/public_gpu_run
```

Custom output paths:

```bash
python scripts/make_public_gpu_report.py \
  --runs benchmarks/public_gpu_run other_run_dir \
  --output-md docs/PUBLIC_GPU_BENCHMARK_REPORT.md \
  --output-json benchmarks/public_gpu_comparison.json
```

## Final Phase 9 checklist

- Environment capture exists.
- Public GPU benchmark runner exists.
- Runbook exists.
- Report generator exists.
- CPU-only test path exists.
- GPU remains optional.
- Experimental CUDA remains opt-in.
- Production generation unchanged.
