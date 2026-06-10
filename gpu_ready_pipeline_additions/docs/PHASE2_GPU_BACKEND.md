# Phase 2 GPU Backend Scaffold

## Goal

Phase 2 adds a small optional backend layer for detecting CuPy/CUDA availability
and moving arrays between CPU and GPU memory. The new backend is intentionally
isolated from the lattice physics code.

## Non-goals

- No CUDA kernels are added.
- No GPU measurement kernels are added.
- No physics algorithms are changed.
- No configuration generation logic is changed.
- No analysis formulas are changed.
- No production default is changed from CPU execution.

## CPU fallback behavior

CPU behavior remains the default. The backend imports CuPy lazily inside helper
functions, so CPU-only environments do not need CuPy installed.

When `use_gpu=False`, the backend always uses NumPy. When `use_gpu=True` but
CuPy is unavailable or no CUDA device is visible, the backend falls back to
NumPy by default. Callers can request strict behavior to raise a clear
`RuntimeError` instead of falling back.

## Backend check

Run the CPU-only smoke check with:

```bash
python scripts/check_gpu_backend.py
```

This prints the Python version, NumPy version, CuPy import status, CUDA device
availability, device count, device name, memory information, and tiny CPU/GPU
array operation results. On CPU-only systems, the script prints a clear fallback
message and exits successfully.

Run the GPU-required smoke check with:

```bash
python scripts/check_gpu_backend.py --strict
```

Strict mode exits nonzero if CuPy cannot import or no CUDA device is available.

## Pytest usage

Run the backend tests with:

```bash
pytest tests/test_gpu_backend.py
```

The test file uses tiny arrays only. CPU fallback tests always run, while
GPU-specific assertions are skipped when CuPy cannot import or no CUDA device is
visible. CuPy remains optional and is not required for the default test path.

## Testing on public GPUs

Public GPU services such as Google Colab, Kaggle Notebooks, RunPod, and Vast.ai
can be used to check that the optional backend detects CUDA correctly. These
steps only test backend readiness; they do not accelerate the physics code yet.

First confirm that the runtime has a visible NVIDIA GPU:

```bash
nvidia-smi
```

Install a CuPy wheel that matches the CUDA runtime exposed by the service. For
CUDA 12.x, use `cupy-cuda12x`; for CUDA 13.x, use `cupy-cuda13x`. Do not install
multiple CuPy packages at the same time. CPU-only development does not need
CuPy, and CuPy is not part of the default dependency path.

### Google Colab

Select a GPU runtime, then run:

```bash
nvidia-smi
pip install cupy-cuda12x
python scripts/check_gpu_backend.py
python scripts/check_gpu_backend.py --strict
python scripts/gpu_smoke_test.py --use-gpu
pytest tests/test_gpu_backend.py
```

Adjust the CuPy package if Colab changes the CUDA runtime version.

### Kaggle Notebooks

Enable GPU acceleration in the notebook settings, then run:

```bash
nvidia-smi
pip install cupy-cuda12x
python scripts/check_gpu_backend.py
python scripts/check_gpu_backend.py --strict
python scripts/gpu_smoke_test.py --use-gpu
pytest tests/test_gpu_backend.py
```

Adjust the CuPy package to match the CUDA version reported by the runtime.

### Generic Linux GPU machine

On a Linux machine with NVIDIA drivers and Python available, run:

```bash
nvidia-smi
python scripts/check_gpu_backend.py
python scripts/gpu_smoke_test.py
```

If CUDA is visible and you want a required-GPU check, install the matching CuPy
wheel, then run:

```bash
python scripts/check_gpu_backend.py --strict
python scripts/gpu_smoke_test.py --use-gpu --strict
pytest tests/test_gpu_backend.py
```

## Scope note

Phase 2 does not accelerate physics yet. Configuration generation is untouched,
and existing physics code paths continue to run on CPU by default.

## Final Phase 2 checklist

- CPU fallback works.
- GPU detection works when a GPU is present.
- Tests pass on CPU-only machines.
- No physics code was changed.
- No files inside `archive/` were touched.
- Phase 3 can now add GPU measurement paths.
