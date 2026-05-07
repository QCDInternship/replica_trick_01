"""End-to-end tadpole-improvement consistency checks."""

import glob
import os
import re
import sys

import numpy as np

import action_v_cfg
import gauge_latticeqcd as glqcd


action = "W"
Nt, Nx, Ny, Nz = 6, 6, 6, 6
beta = 5.7
run_number = 1
alpha = 0
ensemble = 0
plaquette_tol = 5e-3
scaling_rtol = 5e-3


def print_result(label, passed, detail):
    marker = "PASS" if passed else "FAIL"
    print(f"[{marker}] {label}: {detail}")


def load_cached_u0():
    cache_file = f"u0_cache_{Nt}x{Nx}x{Ny}x{Nz}_b{int(beta*100)}.txt"
    if not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0:
        raise FileNotFoundError(f"missing or empty cache file {cache_file}")
    with open(cache_file, "r") as f:
        return float(f.read().strip()), cache_file


def config_number(path):
    match = re.search(r"config_(\d+)_U1\.npy$", os.path.basename(path))
    if match is None:
        return -1
    return int(match.group(1))


def find_latest_config():
    stem = f"Replica_{action}_{Nt}x{Nx}x{Ny}x{Nz}_b{int(beta*100)}"
    candidates = [
        os.path.join(
            f"RunReplica{run_number}_tadpole",
            f"{stem}_alpha{alpha}_ensemble{ensemble}",
        ),
        os.path.join(
            f"RunReplica{run_number}",
            f"{stem}_tadpole_alpha{alpha}_ensemble{ensemble}",
        ),
        os.path.join(
            f"RunReplica{run_number}",
            f"{stem}_alpha{alpha}_ensemble{ensemble}",
        ),
    ]

    for directory in candidates:
        files = glob.glob(os.path.join(directory, "config_*_U1.npy"))
        if files:
            return max(files, key=config_number), directory

    raise FileNotFoundError(
        "no config_*_U1.npy files found in: " + ", ".join(candidates)
    )


def action_with_u0(U, u0_value):
    old_u0 = action_v_cfg.u0
    try:
        action_v_cfg.u0 = u0_value
        return float(action_v_cfg.calc_S(U))
    finally:
        action_v_cfg.u0 = old_u0


def main():
    all_passed = True

    try:
        u0_cache, cache_file = load_cached_u0()
        print_result("load u0 cache", True, f"{cache_file}, u0 = {u0_cache:.8f}")
    except Exception as exc:
        print_result("load u0 cache", False, str(exc))
        return 1

    try:
        config_path, config_dir = find_latest_config()
        cfg = config_number(config_path)
        print_result(
            "load latest thermalised config",
            True,
            f"{config_path} (cfg={cfg}, directory={config_dir})",
        )
        U = np.load(config_path)
    except Exception as exc:
        print_result("load latest thermalised config", False, str(exc))
        return 1

    plaquette = float(glqcd.fn_average_plaquette(U))
    u0_from_plaq = float(plaquette ** 0.25)
    plaquette_diff = abs(u0_from_plaq - u0_cache)
    plaquette_passed = plaquette_diff < plaquette_tol
    all_passed = all_passed and plaquette_passed
    print_result(
        "bare plaquette self-consistency",
        plaquette_passed,
        (
            f"P = {plaquette:.8f} (dimensionless), P**1/4 = {u0_from_plaq:.8f}, "
            f"cached u0 = {u0_cache:.8f}, |delta| = {plaquette_diff:.8f}, "
            f"tol = {plaquette_tol:.8f}"
        ),
    )

    action_cache = action_with_u0(U, u0_cache)
    action_bare = action_with_u0(U, 1.0)
    expected_factor = (1.0 / u0_cache) ** 4
    action_ratio = action_cache / action_bare if action_bare != 0.0 else np.nan
    additive_constant = beta * Nt * Nx * Ny * Nz * 6.0
    plaquette_part_cache = action_cache - additive_constant
    plaquette_part_bare = action_bare - additive_constant
    plaquette_ratio = (
        plaquette_part_cache / plaquette_part_bare
        if plaquette_part_bare != 0.0
        else np.nan
    )
    scaling_passed = (
        np.isfinite(plaquette_ratio)
        and abs(plaquette_ratio - expected_factor) <= scaling_rtol * abs(expected_factor)
    )
    all_passed = all_passed and scaling_passed
    print_result(
        "action with cached u0",
        True,
        f"S(u0={u0_cache:.8f}) = {action_cache:.8f} (dimensionless lattice action)",
    )
    print_result(
        "action with u0=1",
        True,
        f"S(u0=1.00000000) = {action_bare:.8f} (dimensionless lattice action)",
    )
    print_result(
        "raw action ratio",
        True,
        (
            f"S(u0_cache)/S(u0=1) = {action_ratio:.8f}; "
            f"raw totals include the existing additive Wilson constant"
        ),
    )
    print_result(
        "u0**4 plaquette-term scaling",
        scaling_passed,
        (
            f"(S(u0_cache)-const)/(S(u0=1)-const) = {plaquette_ratio:.8f}; "
            f"(1/u0_cache)**4 = {expected_factor:.8f}; rtol = {scaling_rtol:.8f}"
        ),
    )

    if all_passed:
        print("[PASS] tadpole check summary: all required checks passed")
        return 0

    print("[FAIL] tadpole check summary: one or more required checks failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
