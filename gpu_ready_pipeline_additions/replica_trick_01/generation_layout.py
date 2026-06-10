"""Data-layout helpers for future GPU generation preparation.

This module validates array shapes used by the existing CPU generation path. It
does not implement GPU generation or alter production generation behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np


GAUGE_FIELD_TRAILING_SHAPE = (4, 3, 3)
PROPOSAL_MATRIX_TRAILING_SHAPE = (3, 3)


def expected_gauge_field_shape(Nt: int, Nx: int, Ny: int, Nz: int) -> tuple[int, int, int, int, int, int, int]:
    """Return the active gauge-field shape convention."""

    _validate_positive_dimensions(Nt, Nx, Ny, Nz)
    return (Nt, Nx, Ny, Nz, *GAUGE_FIELD_TRAILING_SHAPE)


def expected_proposal_matrix_shape(n_matrices: int) -> tuple[int, int, int]:
    """Return the active proposal matrix container shape convention."""

    if n_matrices < 1:
        raise ValueError("n_matrices must be at least 1")
    return (n_matrices, *PROPOSAL_MATRIX_TRAILING_SHAPE)


def make_synthetic_gauge_field(
    Nt: int,
    Nx: int,
    Ny: int,
    Nz: int,
    dtype: Any = np.complex128,
) -> np.ndarray:
    """Create a tiny identity-link gauge field for layout readiness checks."""

    U = np.zeros(expected_gauge_field_shape(Nt, Nx, Ny, Nz), dtype=dtype)
    U[...] = np.eye(3, dtype=dtype)
    return U


def make_synthetic_proposal_matrices(n_matrices: int, dtype: Any = np.complex128) -> np.ndarray:
    """Create identity proposal matrices for layout readiness checks."""

    proposals = np.zeros(expected_proposal_matrix_shape(n_matrices), dtype=dtype)
    proposals[...] = np.eye(3, dtype=dtype)
    return proposals


def validate_gauge_field_layout(
    U: Any,
    Nt: int,
    Nx: int,
    Ny: int,
    Nz: int,
) -> None:
    """Validate gauge-field shape and complex dtype."""

    array = np.asarray(U)
    expected_shape = expected_gauge_field_shape(Nt, Nx, Ny, Nz)
    if array.shape != expected_shape:
        raise ValueError(f"gauge field shape {array.shape} does not match {expected_shape}")
    if not np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("gauge field must have a complex dtype")


def ensure_gauge_field_layout(U: Any) -> np.ndarray:
    """Return ``U`` as a NumPy array after validating the active gauge layout."""

    array = np.asarray(U)
    if array.ndim != 7:
        raise ValueError("gauge field must have shape (Nt, Nx, Ny, Nz, 4, 3, 3)")
    nt, nx, ny, nz = array.shape[:4]
    validate_gauge_field_layout(array, nt, nx, ny, nz)
    return array


def validate_proposal_matrix_layout(proposals: Any, n_matrices: int) -> None:
    """Validate proposal matrix container shape and complex dtype."""

    array = np.asarray(proposals)
    expected_shape = expected_proposal_matrix_shape(n_matrices)
    if array.shape != expected_shape:
        raise ValueError(f"proposal matrix shape {array.shape} does not match {expected_shape}")
    if not np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("proposal matrices must have a complex dtype")


def describe_array_layout(array: Any) -> dict[str, Any]:
    """Return shape, dtype, contiguity, and memory information for an array."""

    arr = np.asarray(array)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "c_contiguous": bool(arr.flags.c_contiguous),
        "bytes": int(arr.nbytes),
    }


def _validate_positive_dimensions(Nt: int, Nx: int, Ny: int, Nz: int) -> None:
    """Validate positive lattice dimensions."""

    for name, value in (("Nt", Nt), ("Nx", Nx), ("Ny", Ny), ("Nz", Nz)):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
