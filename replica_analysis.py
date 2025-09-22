"""
Entanglement entropy in SU(3) lattice QCD via the Replica Lattice Method

Author: Lukas Wystemp - University of Manchester - Yr3 Physics Summer Project 2025

Depricated script: Use Replica_generate_multiprocessing.py instead.

"""

import numpy as np
import os
import gauge_latticeqcd as glqcd
import lattice_collection as lc
import action_v_cfg
import matplotlib.pyplot as plt

Nx = 4
Ny = 4
Nz = 4
Nt = 4
action = 'W'
beta = 5.7
u0 = 1.0
Nstart = 40
Nend = 49


min_ensemble = 0
max_ensemble = 1
ensemble_step = 1

run_number = 4

base_dir = f"RunReplica{run_number}/Replica_{action}_{Nt}x{Nx}x{Ny}x{Nz}_b{int(beta * 100)}"

S_ensemble_diffs = np.array([])
S_alpha_diffs = np.array([])

alphas = [0, 0.2, 0.4]
for alpha in alphas:
    for ensemble in np.arange(min_ensemble, max_ensemble + ensemble_step, ensemble_step):
        dir_name = f"{base_dir}_alpha{alpha}_ensemble{ensemble}"
        collection_1, collection_2 = lc.fn_replica_lattice_collection(Nt, Nx, Ny, Nz, beta, start=Nstart, end=Nend, path=dir_name) 
        actions_1 = []
        actions_2 = []
        for Ncfg in range(Nstart, Nend):
            Ncfg = Ncfg - Nstart
            U1 = np.array(collection_1[Ncfg], dtype=np.complex128)
            S1 = action_v_cfg.calc_S(U1)
            actions_1.append(S1)

            U2 = np.array(collection_2[Ncfg], dtype=np.complex128)
            S2 = action_v_cfg.calc_S(U2)
            actions_2.append(S2)
        
        S_diff = np.array(actions_2) - np.array(actions_1)
        S_ensemble_diffs = np.append(S_ensemble_diffs, S_diff)
        print(f"Alpha: {alpha}, Ensemble: {ensemble}, Action Difference: {S_diff}")
    S_ensemble_avg = np.mean(S_ensemble_diffs)
    print(f"Average action difference for alpha {alpha}: {S_ensemble_avg}")
    S_alpha_diffs = np.append(S_alpha_diffs, S_ensemble_avg)

plt.figure(figsize=(10, 6))
plt.plot(alphas, S_alpha_diffs, marker='o')
plt.ylabel(r'$\langle S_{L+1} - S_{L} \rangle$')
plt.xlabel(r'$\alpha$')
plt.title('Average Action Difference vs Alpha')
plt.grid()