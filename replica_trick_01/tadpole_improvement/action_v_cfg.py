import os, sys, string
import numba_compat as numba
import numpy as np
import tools_v1 as tool
import gauge_latticeqcd as gl
import params
import lattice_collection as lc


### Script to calculate the evolution of the action as a function of Monte Carlo time
### calc_S uses the tadpole cache produced by Modified_Replica_generate_multiprocessing.py.
### Regenerate that cache by re-running the driver with USE_TADPOLE = True.
Nx = 6
Ny = 6
Nz = 6
Nt = 6
action = 'W'
beta = 5.7
Nstart = 0
Nend = 7

u0_cache_file = f"u0_cache_{Nt}x{Nx}x{Ny}x{Nz}_b{int(beta*100)}.txt"


def load_u0_cache(warn=False):
    if os.path.exists(u0_cache_file) and os.path.getsize(u0_cache_file) > 0:
        with open(u0_cache_file, "r") as f:
            return float(f.read().strip())

    if warn:
        print(
            f"WARNING: missing tadpole cache {u0_cache_file}; "
            "using u0 = 1.0 and disabling tadpole improvement for this run.",
            file=sys.stderr,
        )
    return 1.0


u0 = load_u0_cache()

#@numba.njit
def calc_S(U):
    Nt, Nx, Ny, Nz = len(U), len(U[0]), len(U[0, 0]), len(U[0, 0, 0])
    S = 0.
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):

                    S += gl.fn_eval_point_S(U, t, x, y, z, beta, u0=u0)

                #end z
            #end y
        #end x
    #end t
    return S

"""
def calc_S_replica(U):
    Nt, Nx, Ny, Nz = len(U), len(U[0]), len(U[0, 0]), len(U[0, 0, 0])
    S = 0.
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    txyz = [t, x, y, z]
                    S += gl.ReplicaLattice.eval_point_S_replica(U, txyz, beta, u0=1.)

                #end z
            #end y
        #end x
    #end t
    return S

def calc_S_int(Nt, Nx, Ny, Nz, beta, Nstart, Nend, dir_name, alpha=0.2):
    collection_1, collection_2 = lc.fn_replica_lattice_collection(Nt, Nx, Ny, Nz, beta, Nstart, Nend, path=dir_name)
    S_1_collection = [calc_S(np.array(collection_1[Ncfg], dtype=np.complex128)) for Ncfg in range(Nstart, Nend)]
    S_2_collection = [calc_S(np.array(collection_2[Ncfg], dtype=np.complex128)) for Ncfg in range(Nstart, Nend)]
    S_int = (1 - alpha) * np.array(S_1_collection) + alpha * np.array(S_2_collection)
    return S_int
"""

if __name__ == "__main__":
    u0 = load_u0_cache(warn=True)
    dir = './' + action + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(beta * 100)) + '/'
    U_infile = dir + 'link_' + action + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(beta * 100)) + '_'

    ### prepare output file
    outfile = './S_v_cfg_' + str(int(beta * 100)) + '_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_' + action + '.dat'
    fout = open(outfile, 'a')

    fout.write('#1:cfg  2:S\n')
    for Ncfg in range(Nstart, Nend + 1):
        U = np.load(U_infile + str(Ncfg))
        S = calc_S(U)
        fout.write(str(Ncfg) + ' ' + str(S) + '\n' )
    #end Ncfg
    fout.close()
