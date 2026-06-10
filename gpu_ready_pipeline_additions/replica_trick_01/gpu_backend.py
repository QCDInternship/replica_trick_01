"""Optional NumPy/CuPy backend helpers.

This module intentionally keeps CuPy optional. CuPy is imported lazily inside
functions so normal CPU-only use does not require a CUDA Python stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _cupy_unavailable_error() -> RuntimeError:
    """Build a consistent strict-mode error."""

    return RuntimeError("GPU backend requested, but CuPy/CUDA is not available.")


def _import_cupy() -> Any | None:
    """Return the CuPy module when importable, otherwise ``None``."""

    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        return None
    return cp


def is_cupy_installed() -> bool:
    """Return ``True`` if CuPy can be imported."""

    return _import_cupy() is not None


def get_device_count() -> int:
    """Return the number of visible CUDA devices, or zero on failure."""

    cp = _import_cupy()
    if cp is None:
        return 0
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def is_gpu_available() -> bool:
    """Return ``True`` when CuPy imports and at least one CUDA device is visible."""

    return get_device_count() > 0


def get_device_name(device_id: int = 0) -> str | None:
    """Return the CUDA device name for ``device_id``, or ``None`` if unavailable."""

    cp = _import_cupy()
    if cp is None:
        return None
    try:
        if device_id < 0 or device_id >= int(cp.cuda.runtime.getDeviceCount()):
            return None
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props.get("name")
        if isinstance(name, bytes):
            return name.decode("utf-8", errors="replace")
        if name is None:
            return None
        return str(name)
    except Exception:
        return None


def get_array_module(use_gpu: bool = False, strict: bool = False) -> Any:
    """Return NumPy or CuPy for array creation and simple array operations.

    Parameters
    ----------
    use_gpu:
        Request CuPy when available. ``False`` always returns NumPy.
    strict:
        Raise ``RuntimeError`` instead of falling back to NumPy when ``use_gpu``
        is true and CuPy/CUDA is unavailable.
    """

    if not use_gpu:
        return np

    cp = _import_cupy()
    if cp is None or get_device_count() < 1:
        if strict:
            raise _cupy_unavailable_error()
        return np
    return cp


def to_device(x: Any, use_gpu: bool = False, dtype: Any = None, strict: bool = False) -> Any:
    """Return ``x`` as a NumPy or CuPy array according to ``use_gpu``.

    CPU mode always returns a NumPy array view/copy via ``numpy.asarray``. GPU
    mode returns a CuPy array when CuPy/CUDA is available; otherwise it falls
    back to NumPy unless ``strict`` is true.
    """

    xp = get_array_module(use_gpu=use_gpu, strict=strict)
    return xp.asarray(x, dtype=dtype)


def to_cpu(x: Any) -> Any:
    """Return a NumPy representation of ``x`` when it is a CuPy array.

    NumPy arrays, Python scalars, and other ordinary objects are returned
    unchanged except for CuPy arrays, which are copied back to host memory.
    """

    cp = _import_cupy()
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x


def is_gpu_array(x: Any) -> bool:
    """Return ``True`` if ``x`` is a CuPy ndarray."""

    cp = _import_cupy()
    return bool(cp is not None and isinstance(x, cp.ndarray))


def synchronize() -> None:
    """Synchronize the current CUDA device when available.

    This is a no-op on CPU-only systems or when CuPy/CUDA cannot be used.
    """

    cp = _import_cupy()
    if cp is None:
        return
    try:
        if int(cp.cuda.runtime.getDeviceCount()) > 0:
            cp.cuda.Stream.null.synchronize()
    except Exception:
        return


def memory_info(device_id: int = 0) -> dict[str, Any]:
    """Return CUDA memory information for ``device_id``.

    The returned dictionary always contains ``available``, ``device_id``,
    ``free_bytes``, ``total_bytes``, and ``error`` keys.
    """

    info: dict[str, Any] = {
        "available": False,
        "device_id": device_id,
        "free_bytes": None,
        "total_bytes": None,
        "error": None,
    }
    cp = _import_cupy()
    if cp is None:
        info["error"] = "CuPy is not installed or could not be imported."
        return info

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_id < 0 or device_id >= device_count:
            info["error"] = f"CUDA device {device_id} is not visible."
            return info
        with cp.cuda.Device(device_id):
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        info.update(
            {
                "available": True,
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
            }
        )
    except Exception as exc:
        info["error"] = str(exc)
    return info
