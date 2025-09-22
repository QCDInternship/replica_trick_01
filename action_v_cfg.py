import os, sys, string
import numba
import numpy as np
import tools_v1 as tool
import gauge_latticeqcd as gl
import params
import lattice_collection as lc


### Script to calculate the evolution of the action as a function of Monte Carlo time
Nx = 6
Ny = 6
Nz = 6
Nt = 6
action = 'W'
beta = 5.7
u0 = 1.0
Nstart = 0
Nend = 7

#@numba.njit
def calc_S(U):
    Nt, Nx, Ny, Nz = len(U), len(U[0]), len(U[0, 0]), len(U[0, 0, 0])
    S = 0.
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):

                    S += gl.fn_eval_point_S(U, t, x, y, z, beta, u0=1.)

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
