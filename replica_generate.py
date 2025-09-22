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

# User parameters
action = 'W'
Nt, Nx, Ny, Nz = 16, 16, 16, 16
beta = 5.7
Nstart = 0
Nend = 50
Ncfg = Nend - Nstart + 1
Nhits = 10
n = 2
s = 2
u0 = 1.0

run_number = 3

base_dir = f"RunReplica{run_number}"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
    print(f"Created directory: {base_dir}")
else:
    print("Directory exists for beta ", beta)

new_base_dir = f"Replica_{action}_{Nt}x{Nx}x{Ny}x{Nz}_b{int(beta * 100)}"
new_base_dir = os.path.join(base_dir, new_base_dir)

alphas = [0, 0.2, 0.4]
for _, alpha in enumerate(alphas):
    print(f"\nGenerating configurations for alpha = {alpha}...\n")
    for ensemble in range(2):
        print(f"\nProcessing configuration {ensemble}...\n")

        dir_name = new_base_dir + f"_alpha{alpha}_ensemble{ensemble}"

        U = glqcd.ReplicaLattice(Nt, Nx, Ny, Nz, beta, u0, n, s)
        matrices = glqcd.create_su3_set()
        U.markov_chain_sweep_replica(Ncfg, matrices, Nhits, dir_name)



base_dir = f"RunReplica{run_number}/Replica_{action}_{Nt}x{Nx}x{Ny}x{Nz}_b{int(beta * 100)}"

def read_configs(dir_name):
    actions_1 = []
    actions_2 = []
    for Ncfg in range(Nstart, Nend):
        collection_1, collection_2 = lc.fn_replica_lattice_collection(Nt, Nx, Ny, Nz, beta, start=Nstart, end=Nend, path=base_dir + dir_name)

        U = np.array(collection_1[Ncfg], dtype=np.complex128)
        S = action_v_cfg.calc_S(U)
        actions_1.append(S)

        U = np.array(collection_2[Ncfg], dtype=np.complex128)
        S = action_v_cfg.calc_S(U)
        actions_2.append(S)
    
    return np.array(actions_1), np.array(actions_2)


actions_0_0_1, actions_0_0_2 = read_configs("_alpha0_ensemble0")
actions_0_1_1, actions_0_1_2 = read_configs("_alpha0_ensemble1")
actions_01_0_1, actions_01_0_2 = read_configs("_alpha0.2_ensemble0")
actions_01_1_1, actions_01_1_2 = read_configs("_alpha0.2_ensemble1")
actions_02_0_1, actions_02_0_2 = read_configs("_alpha0.4_ensemble0")
actions_02_1_1, actions_02_1_2 = read_configs("_alpha0.4_ensemble1")


S_int_1 = action_v_cfg.calc_S_int(Nt, Nx, Ny, Nz, beta, Nstart, Nend, dir_name=base_dir + "_alpha0_ensemble0")
S_int_2 = action_v_cfg.calc_S_int(Nt, Nx, Ny, Nz, beta, Nstart, Nend, dir_name=base_dir + "_alpha0.2_ensemble1")
S_int_3 = action_v_cfg.calc_S_int(Nt, Nx, Ny, Nz, beta, Nstart, Nend, dir_name=base_dir + "_alpha0.4_ensemble1")

plt.plot(range(Nstart, Nend), actions_0_0_1, marker='o', label=r' S_{L}, $\alpha=0$, e=0', color='blue')
plt.plot(range(Nstart, Nend), actions_0_0_2, marker='x', label=r' $S_{L+\Delta L}$, $\alpha=0$, e=0', color = 'blue')

plt.plot(range(Nstart, Nend), actions_0_1_1, marker='o', label=r' S_{L}, $\alpha=0$, e=1', color='red')
plt.plot(range(Nstart, Nend), actions_0_1_2, marker='x', label=r' $S_{L+\Delta L}$, $\alpha=0$, e=1', color = 'red')

plt.plot(range(Nstart, Nend), actions_01_0_1, marker='o', label=r' S_{L}, $\alpha=0.2$, e=0', color='yellow')
plt.plot(range(Nstart, Nend), actions_01_0_2, marker='x', label=r' $S_{L+\Delta L}$, $\alpha=0.2$, e=0', color = 'yellow')

plt.plot(range(Nstart, Nend), actions_01_1_1, marker='o', label=r' S_{L}, $\alpha=0.2$, e=1', color='green')
plt.plot(range(Nstart, Nend), actions_01_1_2, marker='x', label=r' $S_{L+\Delta L}$, $\alpha=0.2$, e=1', color = 'green')

plt.plot(range(Nstart, Nend), actions_02_0_1, marker='o', label=r' S_{L}, $\alpha=0.4$, e=0', color='purple')
plt.plot(range(Nstart, Nend), actions_02_0_2, marker='x', label=r' $S_{L+\Delta L}$, $\alpha=0.4$, e=0', color = 'purple')

plt.plot(range(Nstart, Nend), actions_02_1_1, marker='o', label=r' S_{L}, $\alpha=0.4$, e=1', color='pink')
plt.plot(range(Nstart, Nend), actions_02_1_2, marker='x', label=r' $S_{L+\Delta L}$, $\alpha=0.4$, e=1', color = 'pink')


plt.plot(range(Nstart, Nend), S_int_1, marker='^', label=r'$S_{\text{int}}$ $\alpha=0$, e=0', color='blue')
plt.plot(range(Nstart, Nend), S_int_2, marker='^', label=r'$S_{\text{int}}$ $\alpha=0.2$, e=1', color='green')
plt.plot(range(Nstart, Nend), S_int_3, marker='^', label=r'$S_{\text{int}}$ $\alpha=0.4$, e=1', color='pink')


plt.xlabel('Configuration index')
plt.ylabel('Wilson action S')
plt.title('Wilson action across configurations')
plt.grid(True)
plt.legend(loc ='upper left', fontsize = 'xx-small')
plt.savefig(f'Wilson_action_{Nt}x{Nx}x{Ny}x{Nz}_b{int(beta * 100)}_alpha=0-0.4_e=0-1.png', dpi=300)
plt.show()