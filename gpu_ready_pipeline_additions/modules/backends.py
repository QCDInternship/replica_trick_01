"""Explicit backend registry for reference and experimental paths.

CPU remains the default and authoritative backend. GPU and CUDA prototype
entries are opt-in only and do not change production generation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Callable

from replica_trick_01 import gpu_backend


CPU = "cpu"
MPI = "mpi"
GPU_MEASURE = "gpu-measure"
CUDA_PROTOTYPE = "cuda-prototype"


@dataclass(frozen=True)
class BackendInfo:
    """Metadata and availability hook for one backend."""

    name: str
    description: str
    experimental: bool
    available: Callable[[], bool]


def _always_available() -> bool:
    return True


def _mpi_available() -> bool:
    return importlib.util.find_spec("mpi4py") is not None


def _gpu_measure_available() -> bool:
    return gpu_backend.is_gpu_available()


def _cuda_prototype_available() -> bool:
    try:
        from replica_trick_01.cuda_generation import cuda_kernels
    except Exception:
        return False
    return cuda_kernels.is_cuda_available()


_BACKENDS: dict[str, BackendInfo] = {
    CPU: BackendInfo(
        name=CPU,
        description="Authoritative NumPy/CPU reference path. This is the default.",
        experimental=False,
        available=_always_available,
    ),
    MPI: BackendInfo(
        name=MPI,
        description="Optional MPI execution path when mpi4py is installed.",
        experimental=False,
        available=_mpi_available,
    ),
    GPU_MEASURE: BackendInfo(
        name=GPU_MEASURE,
        description="Optional CuPy/CUDA measurement backend; generation remains CPU-only.",
        experimental=False,
        available=_gpu_measure_available,
    ),
    CUDA_PROTOTYPE: BackendInfo(
        name=CUDA_PROTOTYPE,
        description="Experimental CUDA-generation prototype path, never production/default.",
        experimental=True,
        available=_cuda_prototype_available,
    ),
}


def backend_names() -> tuple[str, ...]:
    """Return all registered backend names."""

    return tuple(_BACKENDS)


def available_backends() -> tuple[str, ...]:
    """Return registered backend names that are currently available."""

    return tuple(name for name in backend_names() if is_backend_available(name))


def is_backend_available(name: str) -> bool:
    """Return whether ``name`` is registered and available."""

    info = _backend_info(name)
    try:
        return bool(info.available())
    except Exception:
        return False


def require_backend(name: str) -> BackendInfo:
    """Return backend info or raise a clear error if unavailable."""

    info = _backend_info(name)
    if not is_backend_available(name):
        raise RuntimeError(f"backend '{name}' is not available on this machine")
    return info


def backend_description(name: str) -> str:
    """Return a short backend description."""

    return _backend_info(name).description


def is_experimental_backend(name: str) -> bool:
    """Return whether ``name`` is experimental."""

    return _backend_info(name).experimental


def default_backend() -> str:
    """Return the authoritative default backend."""

    return CPU


def _backend_info(name: str) -> BackendInfo:
    """Return backend metadata or raise for unknown names."""

    try:
        return _BACKENDS[name]
    except KeyError as exc:
        known = ", ".join(backend_names())
        raise ValueError(f"unknown backend '{name}'; expected one of: {known}") from exc
