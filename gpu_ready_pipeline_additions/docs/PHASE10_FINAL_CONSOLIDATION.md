# Phase 10 Final Consolidation

## Goal

Phase 10 consolidates the GPU-ready/HPC-ready work into a clean documentation
and repository organization layer. It adds a documentation index, final
structure notes, and cleanup notes for generated outputs.

## Non-goals

- No production configuration-generation changes.
- No changes to `markov_chain_sweep_replica`.
- No changes to physics parameters, proposal generation, Metropolis acceptance,
  update ordering, replica boundary conditions, staple definitions, action
  definitions, tadpole-improvement logic, RNG behavior, or save frequency.
- No new CUDA kernels.
- No GPU or CUDA defaults.
- No production GPU configuration-generation claim.

## Final Repo Status

- CPU generation remains the authoritative physics path.
- Optional GPU measurement/action kernels are available behind explicit flags.
- GPU-generation layout and RNG scaffolding exists.
- CUDA-generation prototypes are isolated under `replica_trick_01/cuda_generation/`.
- Backend selection, metadata labelling, validation gates, and smoke checks are
  documented.
- Public-GPU environment capture, benchmark orchestration, runbook, and report
  generation exist.

## Remaining Limitations

- CUDA configuration generation is not production.
- Checkerboard/red-black ordering is experimental and requires statistical
  validation.
- Public-GPU results depend on runtime availability and small workloads may not
  show speedup.
- Generated benchmark/profile/prototype files should remain untracked.

## Next Steps After Phase 10

- Run public-GPU benchmarks on real GPU environments.
- Compare reports across Colab, Kaggle, and rented GPU machines.
- Use profiling data to decide whether further measurement-kernel work or an
  isolated CUDA-generation prototype is justified.
- Keep validating any accelerated path against the CPU reference and Phase 0
  observables.

## Final Check Commands

Run CPU-safe final checks:

```bash
python scripts/run_final_checks.py --quick --skip-pytest
```

Run CPU-safe pytest subset:

```bash
pytest tests/test_backends.py tests/test_gpu_backend.py tests/test_generation_layout.py tests/test_rng_streams.py
```

Run the full final check wrapper with pytest:

```bash
python scripts/run_final_checks.py --quick
```

## Optional GPU Testing Commands

Request optional GPU checks:

```bash
python scripts/run_final_checks.py --include-gpu
```

Require a GPU:

```bash
python scripts/run_final_checks.py --strict-gpu
```

Include tiny experimental checks:

```bash
python scripts/run_final_checks.py --include-experimental
```

## CI Limitations

The CPU smoke workflow is CPU-only. It does not install CuPy, does not require
an NVIDIA GPU, does not run paid GPU CI, and does not validate production CUDA
generation. GPU and experimental CUDA checks remain manual opt-ins.

## Final Checklist

- Docs index exists.
- Final checks exist.
- CPU-safe tests exist.
- GPU optional.
- CUDA prototype experimental.
- Safe claims documented.
- Production generator unchanged by consolidation phases.

## Final Report Commands

Generate the final GPU-ready report:

```bash
python scripts/make_final_gpu_ready_report.py
```

Review safe claims:

```text
docs/SAFE_CLAIMS.md
```

Review future work:

```text
docs/FUTURE_WORK.md
```
