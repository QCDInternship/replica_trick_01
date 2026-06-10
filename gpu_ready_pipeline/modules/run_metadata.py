"""Run metadata helpers for backend-labelled outputs.

These helpers label CPU reference, optional measurement, MPI, and experimental
prototype outputs without changing production generation or validation
tolerances.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np

from replica_trick_01 import backends, gpu_backend


EXPERIMENTAL_WARNING = (
    "EXPERIMENTAL OUTPUT: this is not production, is not physics-validated, "
    "and must not be treated as a production result without validation."
)


def get_git_commit() -> str | None:
    """Return the current git commit hash if available."""

    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def get_environment_metadata(include_gpu: bool = True) -> dict[str, Any]:
    """Return Python, NumPy, and optional GPU environment metadata."""

    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    if include_gpu:
        metadata.update(
            {
                "cupy_available": gpu_backend.is_cupy_installed(),
                "gpu_available": gpu_backend.is_gpu_available(),
                "gpu_name": gpu_backend.get_device_name(),
            }
        )
    return metadata


def make_run_metadata(
    backend: str,
    parameters: dict[str, Any],
    experimental: bool = False,
    validation_status: str | None = None,
) -> dict[str, Any]:
    """Create backend-labelled run metadata."""

    backend_experimental = _backend_is_experimental(backend) or bool(experimental)
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        **get_environment_metadata(include_gpu=_include_gpu_metadata(backend)),
        "backend": backend,
        "experimental": backend_experimental,
        "lattice_dimensions": _lattice_dimensions(parameters),
        "beta": parameters.get("beta"),
        "alpha": parameters.get("alpha"),
        "Nhits": parameters.get("Nhits", parameters.get("nhits")),
        "seed": parameters.get("seed"),
        "output_directory": str(parameters.get("output_directory", parameters.get("output_dir", ""))),
        "validation_status": validation_status,
        "warning": EXPERIMENTAL_WARNING if backend_experimental else None,
        "parameters": dict(parameters),
    }
    return metadata


def write_run_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    """Write run metadata JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_run_metadata(path: str | Path) -> dict[str, Any]:
    """Load run metadata JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _backend_is_experimental(name: str) -> bool:
    try:
        return backends.is_experimental_backend(name)
    except ValueError:
        return False


def _include_gpu_metadata(name: str) -> bool:
    return name in {backends.GPU_MEASURE, backends.CUDA_PROTOTYPE, "cuda-prototype"}


def _lattice_dimensions(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "nt": parameters.get("nt"),
        "nx": parameters.get("nx"),
        "ny": parameters.get("ny"),
        "nz": parameters.get("nz"),
    }
