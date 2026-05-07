"""Small compatibility layer for optional numba acceleration."""

try:
    import numba as _numba
except ModuleNotFoundError:
    _numba = None


def njit(*args, **kwargs):
    """Return numba.njit when available, otherwise a no-op decorator."""
    if _numba is not None:
        return _numba.njit(*args, **kwargs)

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator
