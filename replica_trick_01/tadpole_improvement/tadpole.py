"""Lepage-Mackenzie tadpole-improvement helpers.

The Lepage-Mackenzie prescription determines the mean-link factor from the
bare average plaquette, u0 = <(1/3) Re Tr U_plaq>**(1/4), then uses that u0
to rescale gauge-action loop terms self-consistently. See G. P. Lepage and
P. B. Mackenzie, Phys. Rev. D 48, 2250 (1993).
"""

import logging
import os

import numpy as np

import gauge_latticeqcd as glqcd


__all__ = ["measure_u0_from_plaquette", "solve_u0_selfconsistent"]


def measure_u0_from_plaquette(U) -> float:
    """Return u0 from the bare average plaquette of a gauge configuration."""
    plaquette = glqcd.fn_average_plaquette(U)
    return float(plaquette ** 0.25)


def solve_u0_selfconsistent(
    Nt,
    Nx,
    Ny,
    Nz,
    beta,
    n,
    s,
    n_therm=400,
    n_measure=100,
    n_iter=8,
    tol=1e-3,
    u0_init=1.0,
    Nhits=10,
    alpha=0.0,
    ensemble_tag="u0solve",
    base_dir="u0_calibration",
) -> float:
    """Iteratively solve for the tadpole factor u0."""
    del n, s
    if alpha != 0.0:
        logging.warning("u0 calibration ignores alpha=%s and uses plain Wilson geometry.", alpha)

    dir_name = os.path.join(base_dir, ensemble_tag)
    os.makedirs(dir_name, exist_ok=True)
    u0 = float(u0_init)
    matrices = glqcd.create_su3_set()

    for i in range(n_iter):
        u0_old = u0

        # alpha=0 in ReplicaLattice still uses the xcutoff_1 replica temporal
        # boundary, so calibration falls back to the non-replica lattice class.
        lattice = glqcd.lattice(Nt, Nx, Ny, Nz, beta, u0_old)

        if n_therm > 0:
            lattice.markov_chain_sweep(
                Ncfg=n_therm + 1,
                matrices=matrices,
                initial_cfg=0,
                save_name=dir_name,
                Nhits=Nhits,
                action="W",
            )

        plaquettes = []
        for _ in range(n_measure):
            lattice.markov_chain_sweep(
                Ncfg=2,
                matrices=matrices,
                initial_cfg=0,
                save_name=dir_name,
                Nhits=Nhits,
                action="W",
            )
            plaquettes.append(lattice.average_plaquette())

        plaquette_avg = float(np.mean(plaquettes))
        u0_new = float(plaquette_avg ** 0.25)
        print(i, u0_old, plaquette_avg, u0_new)

        if abs(u0_new - u0_old) < tol:
            return u0_new

        u0 = u0_new

    logging.warning(
        "%s did not converge after %d iterations; returning u0=%s",
        ensemble_tag,
        n_iter,
        u0,
    )
    return u0
