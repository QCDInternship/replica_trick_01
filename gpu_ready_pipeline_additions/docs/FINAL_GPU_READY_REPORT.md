# Final GPU-Ready Report

## Executive Summary

The repository now has a CPU-preserving GPU-readiness pipeline. The CPU reference generator remains authoritative and default. Optional GPU measurement/action paths, backend checks, validation gates, public-GPU benchmarking tools, and experimental CUDA-generation scaffolding are documented and tested with CPU-safe checks.

No production GPU configuration generation is claimed.

## What Remained Unchanged

- Production configuration generation.
- `markov_chain_sweep_replica`.
- Physics parameters, proposal generation, Metropolis acceptance, update ordering, replica boundary conditions, action definitions, tadpole-improvement logic, RNG behavior, and save frequency.
- CPU reference paths.

## CPU Reference Status

CPU remains the default authoritative physics path. Final checks status: `passed`.

## HPC/CPU Baseline Status

CPU/HPC baseline documentation and bounded execution structure are documented in Phase 1. MPI remains optional and availability-dependent.

## GPU Backend Status

GPU backend detection is optional. CPU-only machines pass checks without CuPy. Public GPU summary reports GPU available as `False`.

## GPU Measurement/Action Kernel Status

Measurement/action kernels operate on saved or synthetic configurations and preserve CPU reference comparison. GPU timing data is `not run` in current benchmark artifacts.

## GPU-Generation Preparation Status

Generation layout helpers and explicit RNG stream scaffolding exist and are CPU-safe. They do not implement CUDA generation.

## Experimental CUDA-Generation Prototype Status

CUDA-generation prototype code is isolated under `replica_trick_01/cuda_generation/`. It is experimental, opt-in, and not production generation.

## Public GPU Benchmarking Status

Public-GPU environment capture, benchmark runner, runbook, and report generator exist. Missing benchmark data is listed below rather than fabricated.

## Validation Policy

Strict numerical validation is required for deterministic kernels and measurements. Statistical validation is required for changed Markov-chain trajectories.

## Feature Status Table

| Feature | Status | Default? | Requires GPU? | Physics risk | Validation status |
| --- | --- | --- | --- | --- | --- |
| CPU reference generator | implemented | yes | no | low/reference | validated by CPU smoke |
| bounded multiprocessing | documented/available | no | no | medium operational | Phase 1 documented |
| MPI driver | optional | no | no | medium operational | availability-dependent |
| GPU backend detection | implemented | no | optional | low | validated by CPU smoke |
| GPU measurement path | optional | no | optional | low on fixed configs | CPU comparison required |
| action/plaquette GPU backend | optional | no | optional | low on fixed configs | strict numerical checks |
| generation layout helpers | implemented | no | no | low/scaffold | validated by CPU smoke |
| RNG-stream scaffolding | implemented | no | no | medium if wired into generation | validated by CPU smoke |
| CUDA-generation prototype | experimental | no | optional | high | not physics-validated |
| public GPU benchmark runner | implemented | no | optional | low/reporting | CPU fallback smoke run |

## Benchmark Data

- All expected current benchmark summary files were found.

## Known Limitations

- CUDA configuration generation is not production.
- Public GPU timings may be missing on CPU-only machines.
- Tiny benchmarks may not predict production performance.
- Checkerboard/red-black trajectories require statistical validation before physics claims.

## Future Work

- Validate larger lattices.
- Run public GPU benchmarks on actual GPU services.
- Improve GPU measurement vectorisation.
- Profile production-size workloads.
- Derive and validate checkerboard update schemes.
- Compare autocorrelation and thermalisation behavior.
- Investigate real CUDA generation only after validation.

## Safe Claim Summary

Extended an SU(3) replica-lattice QCD codebase with a CPU-preserving GPU-readiness pipeline, including profiling, bounded multiprocessing/MPI job structure, optional CuPy-based measurement kernels, GPU-backend validation, and experimental CUDA-generation scaffolding.

## Claims Not Supported

- Production GPU configuration generation.
- Full CUDA Markov-chain sweeps.
- Full GPU lattice-QCD simulation.
- Physics-equivalent CUDA generator without statistical validation.
