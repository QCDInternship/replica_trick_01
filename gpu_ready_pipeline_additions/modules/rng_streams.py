"""Deterministic RNG stream helpers for future GPU generation work.

These helpers pre-generate proposal indices and Metropolis uniform random
numbers into explicit NumPy arrays. They are not wired into production
generation, so the existing RNG sequencing remains the default path.
"""

from __future__ import annotations

from typing import Any

import numpy as np


STREAM_PROPOSAL_INDICES = "proposal_indices"
STREAM_ACCEPTANCE_UNIFORMS = "acceptance_uniforms"


def make_generation_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy Generator for deterministic generation stream scaffolding."""

    return np.random.default_rng(seed)


def generate_proposal_indices(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    n_matrices: int,
) -> np.ndarray:
    """Generate integer proposal indices with values in ``[0, n_matrices)``."""

    if n_matrices < 1:
        raise ValueError("n_matrices must be at least 1")
    return rng.integers(0, n_matrices, size=shape, dtype=np.int64)


def generate_acceptance_uniforms(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Generate Metropolis acceptance uniforms with values in ``[0.0, 1.0)``."""

    return rng.random(size=shape, dtype=np.float64)


def make_generation_random_streams(
    Nt: int,
    Nx: int,
    Ny: int,
    Nz: int,
    Nhits: int,
    n_matrices: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Create proposal-index and acceptance-uniform streams for one sweep shape."""

    _validate_positive_dimensions(Nt, Nx, Ny, Nz, Nhits)
    if n_matrices < 1:
        raise ValueError("n_matrices must be at least 1")

    rng = make_generation_rng(seed)
    stream_shape = (Nt, Nx, Ny, Nz, 4, Nhits)
    streams = {
        STREAM_PROPOSAL_INDICES: generate_proposal_indices(rng, stream_shape, n_matrices),
        STREAM_ACCEPTANCE_UNIFORMS: generate_acceptance_uniforms(rng, stream_shape),
    }
    validate_generation_random_streams(streams, Nt, Nx, Ny, Nz, Nhits, n_matrices)
    return streams


def validate_generation_random_streams(
    streams: dict[str, Any],
    Nt: int,
    Nx: int,
    Ny: int,
    Nz: int,
    Nhits: int,
    n_matrices: int,
) -> None:
    """Validate stream keys, shapes, dtypes, and value ranges.

    Raises
    ------
    ValueError
        If the stream dictionary is malformed or values are out of range.
    """

    _validate_positive_dimensions(Nt, Nx, Ny, Nz, Nhits)
    if n_matrices < 1:
        raise ValueError("n_matrices must be at least 1")

    expected_shape = (Nt, Nx, Ny, Nz, 4, Nhits)
    missing = {STREAM_PROPOSAL_INDICES, STREAM_ACCEPTANCE_UNIFORMS} - set(streams)
    if missing:
        raise ValueError(f"missing random stream key(s): {sorted(missing)}")

    proposal_indices = np.asarray(streams[STREAM_PROPOSAL_INDICES])
    acceptance_uniforms = np.asarray(streams[STREAM_ACCEPTANCE_UNIFORMS])

    if proposal_indices.shape != expected_shape:
        raise ValueError(
            f"proposal_indices shape {proposal_indices.shape} does not match {expected_shape}"
        )
    if acceptance_uniforms.shape != expected_shape:
        raise ValueError(
            f"acceptance_uniforms shape {acceptance_uniforms.shape} does not match {expected_shape}"
        )
    if not np.issubdtype(proposal_indices.dtype, np.integer):
        raise ValueError("proposal_indices must have an integer dtype")
    if not np.issubdtype(acceptance_uniforms.dtype, np.floating):
        raise ValueError("acceptance_uniforms must have a floating dtype")
    if proposal_indices.size and proposal_indices.min() < 0:
        raise ValueError("proposal_indices contains negative values")
    if proposal_indices.size and proposal_indices.max() >= n_matrices:
        raise ValueError("proposal_indices contains values >= n_matrices")
    if acceptance_uniforms.size and acceptance_uniforms.min() < 0.0:
        raise ValueError("acceptance_uniforms contains values below 0.0")
    if acceptance_uniforms.size and acceptance_uniforms.max() >= 1.0:
        raise ValueError("acceptance_uniforms contains values >= 1.0")


def describe_generation_random_streams(streams: dict[str, Any]) -> dict[str, Any]:
    """Return shapes, dtypes, min/max values, and memory use for RNG streams."""

    description: dict[str, Any] = {"streams": {}, "total_bytes": 0}
    for name in (STREAM_PROPOSAL_INDICES, STREAM_ACCEPTANCE_UNIFORMS):
        array = np.asarray(streams[name])
        description["streams"][name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": _safe_min(array),
            "max": _safe_max(array),
            "bytes": int(array.nbytes),
        }
        description["total_bytes"] += int(array.nbytes)
    return description


def _validate_positive_dimensions(Nt: int, Nx: int, Ny: int, Nz: int, Nhits: int) -> None:
    """Validate positive lattice and hit dimensions."""

    for name, value in (("Nt", Nt), ("Nx", Nx), ("Ny", Ny), ("Nz", Nz), ("Nhits", Nhits)):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")


def _safe_min(array: np.ndarray) -> int | float | None:
    """Return an array minimum as a Python scalar, or ``None`` for empty arrays."""

    if array.size == 0:
        return None
    return array.min().item()


def _safe_max(array: np.ndarray) -> int | float | None:
    """Return an array maximum as a Python scalar, or ``None`` for empty arrays."""

    if array.size == 0:
        return None
    return array.max().item()
