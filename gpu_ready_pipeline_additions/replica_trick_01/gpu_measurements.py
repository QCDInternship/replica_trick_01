"""Optional backend-aware measurement routines.

These helpers mirror the existing CPU measurement formulas while allowing a
full configuration to be copied to a CuPy device once at measurement startup.
Configuration generation remains CPU-only and untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np

from replica_trick_01 import action_kernel_backend, gpu_backend


def load_configuration_cpu(path: str | Path) -> np.ndarray:
    """Load a saved ``.npy`` lattice configuration as a NumPy array."""

    return np.load(Path(path))


def prepare_configuration(U: Any, use_gpu: bool = False, strict: bool = False) -> Any:
    """Return ``U`` on the requested backend.

    CPU mode always returns a NumPy array. GPU mode copies the full
    configuration to the GPU once when CuPy/CUDA is available. If GPU mode is
    requested but unavailable, this falls back to NumPy unless ``strict`` is
    true.
    """

    if use_gpu and not gpu_backend.is_gpu_available():
        if strict:
            raise RuntimeError("GPU measurement requested, but CuPy/CUDA is not available.")
        warnings.warn(
            "GPU measurement requested, but CuPy/CUDA is unavailable; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
    return gpu_backend.to_device(U, use_gpu=use_gpu, dtype=np.complex128, strict=strict)


def _periodic_link(U: Any, txyz: list[int], direction: int) -> Any:
    """Return a backend array link with periodic boundary conditions."""

    nt, nx, ny, nz = U.shape[:4]
    return U[txyz[0] % nt, txyz[1] % nx, txyz[2] % ny, txyz[3] % nz, direction]


def _move_forward_link(U: Any, txyz: list[int], direction: int) -> tuple[Any, list[int]]:
    """Return the forward link and updated coordinate."""

    link = _periodic_link(U, txyz, direction)
    new_txyz = txyz[:]
    new_txyz[direction] += 1
    return link, new_txyz


def _move_backward_link(U: Any, txyz: list[int], direction: int) -> tuple[Any, list[int]]:
    """Return the backward daggered link and updated coordinate."""

    new_txyz = txyz[:]
    new_txyz[direction] -= 1
    link = _periodic_link(U, new_txyz, direction).conj().T
    return link, new_txyz


def _line_move_forward(xp: Any, U: Any, line: Any, txyz: list[int], direction: int) -> tuple[Any, list[int]]:
    """Multiply ``line`` by the forward link in ``direction``."""

    link, new_txyz = _move_forward_link(U, txyz, direction)
    return xp.dot(line, link), new_txyz


def _line_move_backward(xp: Any, U: Any, line: Any, txyz: list[int], direction: int) -> tuple[Any, list[int]]:
    """Multiply ``line`` by the backward daggered link in ``direction``."""

    link, new_txyz = _move_backward_link(U, txyz, direction)
    return xp.dot(line, link), new_txyz


def _plaquette(xp: Any, U: Any, t: int, x: int, y: int, z: int, mu: int, nu: int) -> Any:
    """Return the plaquette matrix using the existing CPU orientation convention."""

    start_txyz = [t, x, y, z]
    result = xp.eye(3, dtype=U.dtype)
    result, next_txyz = _line_move_forward(xp, U, result, start_txyz, mu)
    result, next_txyz = _line_move_forward(xp, U, result, next_txyz, nu)
    result, next_txyz = _line_move_backward(xp, U, result, next_txyz, mu)
    result, next_txyz = _line_move_backward(xp, U, result, next_txyz, nu)
    return result


def _prepared_array_module(U: Any) -> Any:
    """Return the array module for an already prepared configuration."""

    if gpu_backend.is_gpu_array(U):
        return gpu_backend.get_array_module(use_gpu=True, strict=True)
    return np


def _average_plaquette_prepared(U_backend: Any) -> float:
    """Calculate the average plaquette for an already prepared configuration."""

    xp = _prepared_array_module(U_backend)
    nt, nx, ny, nz = U_backend.shape[:4]
    res = xp.zeros(U_backend.shape[-2:], dtype=np.complex128)

    for t in range(nt):
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    for mu in range(1, 4):
                        for nu in range(mu):
                            res = xp.add(res, _plaquette(xp, U_backend, t, x, y, z, mu, nu))

    if gpu_backend.is_gpu_array(res):
        gpu_backend.synchronize()
    value = xp.trace(res).real / 3.0 / nt / nx / ny / nz / 6.0
    return float(np.asarray(gpu_backend.to_cpu(value)))


def average_plaquette_backend(U: Any, use_gpu: bool = False, strict: bool = False) -> float:
    """Calculate the average plaquette with the existing CPU normalization.

    The public return value is always a Python ``float``. When GPU mode is
    active, only the final scalar is copied back to CPU.
    """

    U_backend = prepare_configuration(U, use_gpu=use_gpu, strict=strict)
    return _average_plaquette_prepared(U_backend)


def _wilson_action_prepared(U_backend: Any, beta: float, u0: float = 1.0) -> float:
    """Calculate the ordinary Wilson action for an already prepared configuration."""

    xp = _prepared_array_module(U_backend)
    nt, nx, ny, nz = U_backend.shape[:4]
    total = 0.0

    for t in range(nt):
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    for mu in range(1, 4):
                        for nu in range(mu):
                            plaquette = _plaquette(xp, U_backend, t, x, y, z, mu, nu)
                            total += 1.0 - xp.real(xp.trace(plaquette)) / 3.0 / u0**4

    if gpu_backend.is_gpu_array(U_backend):
        gpu_backend.synchronize()
    value = beta * total
    return float(np.asarray(gpu_backend.to_cpu(value)))


def wilson_action_backend(
    U: Any,
    beta: float,
    u0: float = 1.0,
    use_gpu: bool = False,
    strict: bool = False,
) -> float:
    """Calculate the ordinary Wilson action using existing CPU conventions."""

    U_backend = prepare_configuration(U, use_gpu=use_gpu, strict=strict)
    return _wilson_action_prepared(U_backend, beta=beta, u0=u0)


def measurement_summary(
    U: Any,
    beta: float,
    u0: float = 1.0,
    use_gpu: bool = False,
    strict: bool = False,
    kernel: str = "reference",
) -> dict[str, Any]:
    """Return a small plaquette/action summary for one configuration."""

    if kernel not in {"reference", "backend"}:
        raise ValueError("kernel must be 'reference' or 'backend'")
    if kernel == "backend":
        summary = action_kernel_backend.action_kernel_summary(
            U,
            beta=beta,
            u0=u0,
            use_gpu=use_gpu,
            strict=strict,
        )
        summary["kernel"] = "backend"
        return summary

    U_backend = prepare_configuration(U, use_gpu=use_gpu, strict=strict)
    used_gpu = gpu_backend.is_gpu_array(U_backend)
    plaquette = _average_plaquette_prepared(U_backend)
    action = _wilson_action_prepared(U_backend, beta=beta, u0=u0)
    return {
        "kernel": "reference",
        "backend": "gpu" if used_gpu else "cpu",
        "use_gpu_requested": bool(use_gpu),
        "gpu_available": gpu_backend.is_gpu_available(),
        "shape": list(U_backend.shape),
        "dtype": str(U_backend.dtype),
        "beta": float(beta),
        "u0": float(u0),
        "average_plaquette": plaquette,
        "wilson_action": action,
    }
