from __future__ import print_function
import numba
import numpy as np
import os
import sys
import lattice_collection as lc
import tools_v1 as tool
import datetime
import params
from tqdm import tqdm

### File with Lattice class to sweep and generate lattices and functions: 
### plaquette, average plaquette, polyakov, planar and non-planar wilson loops, wilson action,
### operator density and operator sum helper functions, topological charge included.


#-------------Measurment code -------------------
### some functions are reproduced here, outside of Lattice class, to be accessible via function call.
### - a bit redundant
def fn_periodic_link(U, txyz, direction):
    Nt, Nx, Ny, Nz = len(U), len(U[0]), len(U[0][0]), len(U[0][0][0])
    return U[txyz[0] % Nt][txyz[1] % Nx][txyz[2] %Ny][txyz[3] % Nz][direction]

def fn_move_forward_link(U, txyz, direction):
    link = fn_periodic_link(U, txyz, direction)
    new_txyz = txyz[:]
    new_txyz[direction] += 1
    return link, new_txyz

def fn_move_backward_link(U, txyz, direction):
    new_txyz = txyz[:]
    new_txyz[direction] -= 1
    link = fn_periodic_link(U, new_txyz, direction).conj().T
    return link, new_txyz

def fn_line_move_forward(U, line, txyz, direction):
    link, new_txyz = fn_move_forward_link(U, txyz, direction)
    new_line = np.dot(line, link)
    return new_line, new_txyz

def fn_line_move_backward(U, line, txyz, direction):
    link, new_txyz = fn_move_backward_link(U, txyz, direction)
    new_line = np.dot(line, link)
    return new_line, new_txyz

### plaquette calculation
def fn_plaquette(U, t, x, y, z, mu, nu):
    Nt = len(U)
    Nx = len(U[0])
    Ny = len(U[0][0])
    Nz = len(U[0][0][0])
    start_txyz = [t,x,y,z]
    result = 1.
    result, next_txyz = fn_line_move_forward(U, 1., start_txyz, mu)
    result, next_txyz = fn_line_move_forward(U, result, next_txyz, nu)
    result, next_txyz = fn_line_move_backward(U, result, next_txyz, mu)
    result, next_txyz = fn_line_move_backward(U, result, next_txyz, nu)    
    return result


### Kogut et al, PRL51 (1983) 869, Quark and gluon latent heats at the deconfinement phase transtion in SU(3) gauge theory
### energy density: \varepsilon = \beta / Nt / Ns^3 { (\sum_{space} 1 - ReTrUUUU /3 ) - (\sum{time} 1 - ReTrUUUU /3 )}
### this is just the leading term
def fn_energy_density(U, beta):
    Nt, Nx, Ny, Nz = map(len, [U, U[0], U[0][0], U[0][0][0]])    
    temporal, spatial = 0., 0.
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    for mu in range(4):
                        for nu in range(mu):
                            plaq = fn_plaquette(U, t, x, y, z, mu, nu)
                            plaq = (np.add(plaq, plaq.conj().T))       # avg both orientations averaged
                            plaq = np.trace(plaq.real) / 3. / 2.       # divide by 3 for su3 and 2 for both orientations
                            if mu == 0 or nu == 0:                     # a temporal plaquette
                                temporal += (1 - plaq) 
                            else:
                                spatial  += (1 - plaq)
    energy_dens = spatial - temporal
    energy_des = energy_dens * beta / Nt / Nx / Ny / Nz
    return energy_dens

### calculate average plaquette for u0 = <P_{\mu\nu}>^0.25
#@numba.njit
def fn_average_plaquette(U):
    Nt, Nx, Ny, Nz = map(len, [U, U[0], U[0][0], U[0][0][0]])
    res = np.zeros(np.shape(U[0,0,0,0,0,:,:]), dtype='complex128')
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    for mu in range(1, 4):
                        for nu in range(mu):
                            res = np.add(res, fn_plaquette(U, t, x, y, z, mu, nu))
    return np.trace(res).real / 3. / Nt / Nx / Ny / Nz / 6.

### Wilson action at a specific point
### S = \sum_x \sum_{\mu > \nu} (1 - 1/3 Re Tr P_{\mu\nu}(x))
###   * P_{\mu\nu}(x) = U_\mu(x) U_\nu(x + \hat\mu) U^\dagger_mu(x + \hat\nu) U^\dagger_\nu(x)
###   * fn_plaquette(U,t,x,y,z,mu,nu) returns the product of links around the plaquette, P_{\mu\nu}(x)
###   * beta = 6 / g^2
def fn_eval_point_S(U, t, x, y, z, beta, u0 = 1.):
    tmp = 0.
    for mu in range(1, 4):  #sum over \mu > \nu spacetime dimensions
        for nu in range(mu):
            tmp += ( 1. - np.real(np.trace( fn_plaquette(U, t, x, y, z, mu, nu) )) / 3. / u0**4 )
    return beta * tmp


### Calculate density for given operator.
### Requires lattice and operator to calculate along with all arguments that need to be passed to operator.
def fn_operator_density(U, operator_function, *args):
    Nt, Nx, Ny, Nz = map(len, [U, U[0], U[0][0], U[0][0][0]])
    tmp = [[[[0 for z in range(Nz)] for y in range(Ny)] for x in range(Nx)] for t in range(Nt)]
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    tmp[t][x][y][z] = operator_function(U, t, x, y, z, *args)
    return tmp


### Calculate sum for given operator over whole lattice.
### Requires lattice and operator to calculate along with all arguments that need to be passed to operator.
def fn_sum_over_lattice(U, operator_function, *args):
    Nt, Nx, Ny, Nz = map(len, [U, U[0], U[0][0], U[0][0][0]])
    sum_lattice = 0.
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    sum_lattice += operator_function(U, t, x, y, z, *args)
    return sum_lattice

### Planar Wilson loop with one dimension in time
def fn_wilson(U, t, x, y, z, mu, R, T):  #mu spatial
    #Imagine 2D loop as ABCD where A, B, C, D are edges. Split to two lines:
    #lower that consists of A*B edges
    #upper that consists of C*D edges
    #need to compute specific coordinate points where each line starts.
    if mu == 0:
        print("Wilson loop RT can not be in time direction.")
        exit()
    
    #need start of edge A and start of edge C. start of the two wilson lines
    pointA = [t, x, y, z] 
    pointC = [t, x, y, z]
    pointC[0] += T
    pointC[mu] += R
    lower = 1.
    upper = 1.
    #multiply in correct order
    for nt in range(T):
        lower, pointA = fn_line_move_forward(U, lower, pointA, 0)
        upper, pointC = fn_line_move_backward(U, upper, pointC, 0)
    for nx in range(R):
        lower, pointA = fn_line_move_forward(U, lower, pointA, mu)
        upper, pointC = fn_line_move_backward(U,upper,  pointC, mu)
    result = np.dot(lower, upper)
    return np.trace(result).real / 3.

### average of Wilson loop
def fn_wilson_average(U, R, T):
    Nt, Nx, Ny, Nz = map(len, [U, U[0], U[0][0], U[0][0][0]])
    sum_wilson = 0.
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    for direc in range(1, 4):
                        sum_wilson += fn_wilson(U, t,x,y,z, direc, R, T)
    return sum_wilson / Nx / Ny / Nz / Nt / 3.

### Wilson at a specific R.
### could be calculated with wilson operator as well
### clearer to have this option too
def fn_wilson_loops_at_r(U, R):
    Nt, Nx, Ny, Nz = map(len, [U, U[0], U[0][0], U[0][0][0]])
    wilson_loops = []
    for T in range(Nt):
        tmp = fn_wilson_average(U, R, T)
        wilson_loops.append(tmp)
    return wilson_loops

### non planar Wilson loop -> needed to get intermediate values of potential
def fn_nonplanar_wilson(U, t, x, y, z, mu, R, rho, S, T):
    #imagine non planar loop in 3D as ABCDEF where A, B, C, D, E, F are the EDGES
    #need to calculate all edges and then multiply in order A*B*C*D*E*F to get correct result
    #need to determine specific coordinate points where loop starts
    if mu == 0 or rho == 0:
        print("Non planar Wilson loop RST can not be in time direction.")
        exit()
    if mu == rho:
        print("Non planar Wilson loop RST can not have same spatial directions R, S.")
        exit()

    #Starting points for each edge. pointA for edge A etc
    pointA = [t,x,y,z]
    
    pointB = [t,x,y,z]
    pointB[0] += T
    
    pointC = [t,x,y,z]
    pointC[0] += T
    pointC[mu] += R

    pointD = [t,x,y,z]
    pointD[0] += T
    pointD[mu] += R
    pointD[rho] += S

    pointE = [t,x,y,z]
    pointE[mu] += R
    pointE[rho] += S

    pointF = [t,x,y,z]
    pointF[mu] += R

    edgeA, edgeB, edgeC, edgeD, edgeE, edgeF = 1., 1., 1., 1., 1., 1.
    for nt in range(T):
        edgeA, pointA = fn_line_move_forward(U, edgeA, pointA, 0)
        edgeD, pointD = fn_line_move_backward(U, edgeD, pointD, 0)
    
    for nr in range(R):
        edgeB, pointB = fn_line_move_forward(U, edgeB, pointB, mu)
        edgeF, pointF = fn_line_move_backward(U, edgeF, pointF,mu )

    for ns in range(S):
        edgeC, pointC = fn_line_move_forward(U, edgeC, pointC, rho)
        edgeE, pointE = fn_line_move_backward(U, edgeE, pointE,rho )

    #create full loop -> careful order required
    loop = np.dot(np.dot(np.dot(np.dot(np.dot(edgeA,
                                              edgeB),
                                              edgeC),
                                              edgeD),
                                              edgeE),
                                              edgeF)
    return np.trace(loop).real / 3.

### same as Wilson for nonplanar
def fn_nonplanar_wilson_average(U, R, S, T):
    Nt, Nx, Ny, Nz = map(len, [U, U[0], U[0][0], U[0][0][0]])
    sum_wilson = 0.
    count = 0 #for averaging
    for t in range(Nt):
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    for mu in range(1, 4): #can not pass time in wilson R, T loop
                        for rho in range(1, 4):
                            if mu != rho:
                                sum_wilson += fn_nonplanar_wilson(U, t,x,y,z, mu, R, rho, S,  T)
                                count += 1
    return sum_wilson / float(count)

#same as wilson for nonplanar
def fn_nonplanar_wilson_loops_at_r(U, R, S):
    Nt, Nx, Ny, Nz = map(len, [U, U[0], U[0][0], U[0][0][0]])
    wilson_loops = []
    for T in range(Nt):
        tmp = fn_nonplanar_wilson_average(U, R, S, T)
        wilson_loops.append(tmp)
    return wilson_loops

### Polyakov loop 
def fn_polyakov(U):
    Nt, Nx, Ny, Nz = map(len, [U, U[0], U[0][0], U[0][0][0]])
    ans = 0.
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                p = U[0][x][y][z][0]
                for t in range(1,Nt):
                    p = np.dot(p, U[t][x][y][z][0])
                ans  = np.add(ans, np.trace(p))
    return ans / Nx / Ny / Nz 

### Polyakov density
def fn_polyakov_atpoint(U, x, y, z):
    Nt, Nx, Ny, Nz = len(U), len(U[0]), len(U[0][0]), len(U[0][0][0])
    line = 1.
    for t in range(Nt):
        line = np.dot(line, U[t][x][y][z][0])
    return np.trace(line)

### topological charge that works with only 6 terms
def fn_topological_charge(U, t, x, y, z):
    F01 = fn_F_munu(U, t, x, y, z, 0, 1)
    F23 = fn_F_munu(U, t, x, y, z, 2, 3)
    F02 = fn_F_munu(U, t, x, y, z, 0, 2)
    F31 = fn_F_munu(U, t, x, y, z, 3, 1)
    F03 = fn_F_munu(U, t, x, y, z, 0, 3)
    F12 = fn_F_munu(U, t, x, y, z, 1, 2)
    result = np.trace( np.dot(F01, F23) + np.dot(F02, F31) + np.dot(F03, F12))
    return result / ( 4. * np.pi**2 )

### antihermitian, traceless version of field strength
def fn_F_munu(U, t, x, y, z, mu, nu):
    Pmunu = fn_plaquette(U, t, x, y, z, mu, nu)
    return -1.0J * (np.subtract(Pmunu, Pmunu.conj().T) - np.trace(np.subtract(Pmunu, Pmunu.conj().T)) / 3.) / 2.


#-------------Generation code -------------------
### function called by multiprocessor in generate script
def generate(beta, u0, action, Nt, Nx, Ny, Nz, startcfg, Ncfg, Nhits, Nmatrix, epsilon, Nu0_step='', Nu0_avg = 10):    
    
    ### loop over (t,x,y,z) and mu and set initial collection of links
    ### Either:
    ###  1. initialize to warm start by using random collection of SU(3) links, or
    ###  2. read in a previously generated configuration and continue with that Markov chain.

    name = action +'_' + str(Nt) + 'x' + str(Nx) + 'x' + str(Ny) + 'x' + str(Nz) + '_b' + str(int(beta * 100))
    aa = tool.fn_a( beta )

    print('simulation parameters:')
    print('      action: ' + action)
    print('Nt,Nx,Ny,Nz = ' + str(Nt) + ',' + str(Nx) + ',' + str(Ny) + ',' + str(Nz))
    print('       beta = ' + str(beta))
    print('         u0 = ' + str(u0))
    print('      Nhits = ' + str(Nhits))
    print('      start = ' + str(startcfg))
    print('     sweeps = ' + str(Ncfg))
    print('          a = ' + str(aa) + ' fm')
    print('        1/a = ' + str(params.hbarc_GeVfm / aa) + ' GeV')
    print('        aNx = ' + str(aa * Nx) + ' fm')
    print('Temperature = ' + str(1000. * params.hbarc_GeVfm / (Nt * aa)) + ' MeV')

    if startcfg == 0:
        U = lattice(Nt, Nx, Ny, Nz, beta, u0)
    else:
        #print(action)
        U = lc.fn_load_configuration(action, Nt, Nx, Ny, Nz, beta, startcfg, "./")
        U = lattice(Nt, Nx, Ny, Nz, beta, u0, U)
    
    print('Continuing from cfg: ', startcfg)
    print('... generating lattices')
    matrices = create_su3_set(epsilon, Nmatrix)
    acceptance = U.markov_chain_sweep(Ncfg, matrices, startcfg, name, Nhits, action, Nu0_step, Nu0_avg)
    print("acceptance:", acceptance)

### Generate SU(2) matrix as described in Gattringer & Lang
def matrix_su2(epsilon = 0.2):
    ### Pauli matrices
    sigma1 = np.array([[0, 1], [1, 0]])
    sigma2 = np.array([[0, -1J], [1J, 0]])
    sigma3 = np.array([[1, 0], [0, -1]])
    r = [0., 0., 0., 0.]
    for i in range(4):
        r[i] = (np.random.uniform(0, 0.5))
    ### normalize
    norm = np.sqrt(r[1]**2 + r[2]**2 + r[3]**2)
    r[1:] = map(lambda x: epsilon*x / norm, r[1:])
    r[0]  = np.sign(r[0]) * np.sqrt(1. - epsilon**2)
    M = np.identity(2, dtype='complex128')
    M = M * r[0]
    M = np.add(1J * r[1] * sigma1, M)
    M = np.add(1J * r[2] * sigma2, M)
    M = np.add(1J * r[3] * sigma3, M)
    return M

### Use SU(2) matrices to generate SU(3) matrix
### From Gattringer & Lang's textbook.
### Need 3 SU(2) matrices for one SU(3) matrix
def matrix_su3(epsilon = 0.2):
    R_su2 = matrix_su2(epsilon)
    S_su2 = matrix_su2(epsilon)
    T_su2 = matrix_su2(epsilon)
    # initialise to identity, need complex numbers from now
    R = np.identity(3, dtype='complex128')
    S = np.identity(3, dtype='complex128')
    T = np.identity(3, dtype='complex128')
    # upper
    R[:2,:2] = R_su2
    # edges
    S[0:3:2, 0:3:2] = S_su2
    # lower
    T[1:,1:] = T_su2
    # create final matrix
    X = np.dot(R, S)
    return np.dot(X, T)

### Create set of SU(3) matrices
### Needs to be large enough to cover SU(3)
def create_su3_set(epsilon = 0.2, tot = 1000):
    matrices = []
    for i in range(tot):
        X = matrix_su3(epsilon)
        matrices.append(X)
        matrices.append(X.conj().T)
    return matrices


### LATTICE CLASS
class lattice():
    ### Lattice initialization.
    ### If U not passed, lattice of identities returned.
    ### Class to avoid use of incorrect initialization and passing a lot of variables
    #@numba.njit
    def __init__(self, Nt, Nx, Ny, Nz, beta, u0, U=None):
        if None == U:
            # initialize to identities
            U = [[[[[np.identity(3, dtype='complex128') for mu in range(4)] for z in range(Nz)] for y in range(Ny)] for x in range(Nx)] for t in range(Nt)]
        # convert to numpy arrays -> significant speed up
        self.U = np.array(U)
        self.beta = beta
        self.u0 = u0
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nt = Nt
        
    ### calculate link imposing periodic boundary conditions
    def periodic_link(self, txyz, direction):
        return self.U[txyz[0] % self.Nt, txyz[1] % self.Nx, txyz[2] % self.Ny, txyz[3] % self.Nz, direction, :, :]
    
    def move_forward_link(self, txyz, direction):
        link = self.periodic_link(txyz, direction)
        new_txyz = txyz[:]
        new_txyz[direction] += 1
        return link, new_txyz

    def move_backward_link(self, txyz, direction):
        new_txyz = txyz[:]
        new_txyz[direction] -= 1
        link = self.periodic_link(new_txyz, direction).conj().T
        return link, new_txyz

    def line_move_forward(self, line, txyz, direction):
        link, new_txyz = self.move_forward_link(txyz, direction)
        new_line = np.dot(line, link)
        return new_line, new_txyz

    def line_move_backward(self, line, txyz, direction):
        link, new_txyz = self.move_backward_link(txyz, direction)
        new_line = np.dot(line, link)
        return new_line, new_txyz

    ###WILSON ACTION staple
    #@numba.njit
    def dS_staple(self, t, x, y, z, mu):
        # depends on current position and U over the lattice
        tmp = np.zeros((3, 3), dtype='complex128')
        for nu in range(4):
            if nu != mu:

                #Determine required points for the calculation of the action
                start_txyz = [t, x, y, z]
                start_txyz[mu] += 1

                ### staple 1
                line1 = 1.
                line1, next_txyz = self.line_move_forward(line1, start_txyz, nu)
                line1, next_txyz = self.line_move_backward(line1, next_txyz, mu)
                line1, next_txyz = self.line_move_backward(line1, next_txyz, nu)
                tmp += line1
                
                ### staple 2
                line2 = 1.
                line2, next_txyz = self.line_move_backward(line2, start_txyz, nu)
                line2, next_txyz = self.line_move_backward(line2, next_txyz, mu)
                line2, next_txyz = self.line_move_forward(line2, next_txyz, nu)
                tmp += line2
        
        return tmp / self.u0**3
    
    ### Improved action with rectangles
    def dS_staple_rectangle(self, t, x, y, z, mu):
        plaquette = np.zeros((3, 3), dtype = 'complex128')
        rectangle = np.zeros((3, 3), dtype = 'complex128')

        #loop through nu different than mu
        for nu in range(4):
            if nu != mu:
                start_txyz = [t, x, y, z]
                start_txyz[mu] += 1

                #positive plaquette
                line = 1.
                line, next_txyz = self.line_move_forward(line, start_txyz, nu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_backward(line, next_txyz, nu)
                plaquette += line
                
                #negative plaquette
                line = 1.
                line, next_txyz = self.line_move_backward(line, start_txyz, nu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_forward(line, next_txyz, nu)
                plaquette += line
                
                #rectangle Right right up left left down (Rrulld)
                #capital is the link that we compute staples around -> NOT INCLUDED IN STAPLE
                #NOTE: easier to draw individually to see what they are
                line = 1. 
                line, next_txyz = self.line_move_forward(line, start_txyz, mu)
                line, next_txyz = self.line_move_forward(line, next_txyz, nu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_backward(line, next_txyz, nu)
                rectangle += line
                
                #rectangle Rulldr
                line = 1.
                line, next_txyz = self.line_move_forward(line, start_txyz, nu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_backward(line, next_txyz, nu)
                line, next_txyz = self.line_move_forward(line, next_txyz, mu)
                rectangle += line

                #Ruuldd
                line = 1.
                line, next_txyz = self.line_move_forward(line, start_txyz, nu)
                line, next_txyz = self.line_move_forward(line, next_txyz, nu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_backward(line, next_txyz, nu)
                line, next_txyz = self.line_move_backward(line, next_txyz, nu)
                rectangle += line

                #Rrdllu
                line = 1.
                line, next_txyz = self.line_move_forward(line, start_txyz, mu)
                line, next_txyz = self.line_move_backward(line, next_txyz, nu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_forward(line, next_txyz, nu)
                rectangle += line
                
                #Rdllur
                line = 1.
                line, next_txyz = self.line_move_backward(line, start_txyz, nu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_forward(line, next_txyz, nu)
                line, next_txyz = self.line_move_forward(line, next_txyz, mu)
                rectangle += line
                
                #Rddluu
                line = 1.
                line, next_txyz = self.line_move_backward(line, start_txyz, nu)
                line, next_txyz = self.line_move_backward(line, next_txyz, nu)
                line, next_txyz = self.line_move_backward(line, next_txyz, mu)
                line, next_txyz = self.line_move_forward(line, next_txyz, nu)
                line, next_txyz = self.line_move_forward(line, next_txyz, nu)
                rectangle += line

        ### Return staple corrected with rectangles
        return (5. * plaquette / self.u0**3 / 9.) - (rectangle / self.u0**5 / 36.)  

    
    ### Difference of action. Gets link, updated link, and staple
    #def deltaS(self, link, updated_link, staple):
    #    change = np.trace(np.dot((updated_link - link), staple))
    #    return -self.beta * np.real(change)


    ### Difference of action at a point for fixed staple. Gets link, updated link, and staple A.
    def deltaS(self, link, updated_link, staple):
        return (-self.beta / 3.0 / self.u0 ) * np.real(np.trace(np.dot( (updated_link - link), staple)))


    #@numba.njit
    def plaquette(self, t, x, y, z, mu, nu):
        Nt, Nx, Ny, Nz = self.Nt, self.Nx, self.Ny, self.Nz
        start_txyz = [t, x, y, z]
        result = 1.
        result, next_txyz = self.line_move_forward(1., start_txyz, mu)
        result, next_txyz = self.line_move_forward(result, next_txyz, nu)
        result, next_txyz = self.line_move_backward(result, next_txyz, mu)
        result, next_txyz = self.line_move_backward(result, next_txyz, nu)
        return result
    
    #@numba.njit
    def average_plaquette(self):
        Nt, Nx, Ny, Nz = self.Nt, self.Nx, self.Ny, self.Nz
        res = np.zeros(np.shape(self.U[0, 0, 0, 0, 0, :, :]), dtype='complex128')
        for t in range(Nt):
            for x in range(Nx):
                for y in range(Ny):
                    for z in range(Nz):
                        for mu in range(1, 4):
                            for nu in range(mu):
                                res = np.add(res, self.plaquette(t, x, y, z, mu, nu))
        return np.trace(res).real / 3. / Nt/ Nx / Ny / Nz / 6.
    
    
    ### Markov chain sweep. Requires: 
    ###   number of cfgs,
    ###   set of matrices to generate update,
    ###   initial cfg,
    ###   save name (if given, otherwise will not save),
    ###   hits per sweep,
    ###   action-> W for Wilson or WR for Wilson with rectangles
    ###            W_T or WR_T for tadpole improvement
    def markov_chain_sweep(self, Ncfg, matrices, initial_cfg=0, save_name='', Nhits=10, action='W', Nu0_step='', Nu0_avg=10):
        ratio_accept = 0.
        matrices_length = len(matrices)
        if save_name:
            output = save_name + '/link_' + save_name + '_'
        
        #if tadpole improving, initialize list of u0 values
        if  action[-1:] == 'T':
            plaquette = []
            u0_values = [self.u0]

        ### loop through number of configurations to be generated
        for i in range(Ncfg - 1):
            print('starting sweep ' + str(i) + ':  ' + str(datetime.datetime.now()))

            ### loop through spacetime dimensions
            for t in range(self.Nt):
                for x in range(self.Nx):
                    for y in range(self.Ny):
                        for z in range(self.Nz):
                            ### loop through directions
                            for mu in range(4):
                                ### check which staple to use
                                if (action == 'W') or (action == 'W_T'):
                                    A =  self.dS_staple(t, x, y, z, mu) #standard Wilson or tadpole improved
                                    #(only difference is in save name of lattice since tadpole improvement is 
                                    #considered when calculating staple)
                                elif (action == 'WR') or (action == 'WR_T'):
                                    A = self.dS_staple_rectangle(t, x, y, z, mu) #improved action with rectangles
                                    #Tadpole improve, else only half of O(a^2) error is cancelled.
                                else:
                                    print("Error: Wrong action name or not implemented.")
                                    sys.exit()
                                ### loop through hits
                                for j in range( Nhits ):
                                    ### get a random SU(3) matrix
                                    r = np.random.randint(0, matrices_length) 
                                    matrix = matrices[r] 
                                    ### create U'
                                    Uprime = np.dot(matrix, self.U[t, x, y, z, mu, :, :])
                                    ### calculate staple
                                    dS = self.deltaS(self.U[t, x, y, z, mu, :, :], Uprime, A)
                                    ### check if U' accepted
                                    if (np.exp(-1. * dS) > np.random.uniform(0, 1)):
                                        self.U[t, x, y, z, mu, :, :] = Uprime
                                        ratio_accept += 1
                                        

            ### Update u0. For better performance, skip every Nu0_step cfgs and append plaquettes to array. 
            ### When the array reaches size Nu0_avg, average to update u0.
            ### Wait 10 iterations from warm start.
            if action[-1:] == 'T' and (i % Nu0_step == 0) and i > 10:
                plaquette.append( self.average_plaquette() )
                if len(plaquette) == Nu0_avg:
                    u0_prime = ( np.mean( plaquette ) )**0.25
                    print("u0 for lattice ", self.beta, " to be updated. Previous: ", self.u0, ". New: ", u0_prime)
                    self.u0 = u0_prime           #update u0
                    u0_values.append( u0_prime ) #create array of u0 values
                    plaquette = []               #clear array

            ### save if name given
            if (save_name):
                idx = int(i) + initial_cfg
                #print(int( idx ))
                output_idx = output + str(int( idx ))
                file_out = open(output_idx, 'wb')
                np.save(file_out, self.U)  #NOTE: np.save without opening first appends .npy
                sys.stdout.flush()
        
        ratio_accept = float(ratio_accept) / Ncfg / self.Nx / self.Ny / self.Nz / self.Nt / 4. / Nhits
        if action[-1:] == 'T':
            print("u0 progression: ", u0_values)
        return ratio_accept

"""
Entanglement entropy in SU(3) lattice QCD via the Replica Lattice Method

Author: Lukas Wystemp - University of Manchester - Yr3 Physics Summer Project 2025

Description: Replica lattice class with functions to handle periodic boundary conditions
given n and s. Rewritten functions to calculate action with new boundary conditions and markov chain sweep.

Uses dS_int = (1 - alpha) dS + alpha dS to determine changes. 

Recommended reading:
- Itou, Etsuko, et al. “Entanglement in Four-Dimensional SU(3) Gauge Theory.” 
    Progress of Theoretical and Experimental Physics, vol. 2016, no. 6, June 2016, p. 061B01, 
    doi:10.1093/ptep/ptw050.

- Jokela, Niko. Disentangling the Gravity Dual of Yang-Mills Theory. 2023.
"""
class ReplicaLattice:
    def __init__(self, Nt, Nx, Ny, Nz, beta, u0, n, s, U1=None):
        identity_matrix = np.identity(3, dtype='complex128')
        if U1 is None:
            U1 = np.tile(identity_matrix, (Nt, Nx, Ny, Nz, 4, 1, 1))
        self.U1 = U1
        self.beta = beta
        self.Nt = Nt
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.u0 = u0
        self.n = n # number of replicas, recommended n = 2
        self.s = s # distance of A

    def is_a(self, txyz, xcutoff):
        if txyz[1] >= xcutoff: 
            return True
        return False

    def get_part_bounds(self):
        """Returns a list of (start, end) tuples defining each partition."""
        if self.Nt % self.n != 0:
            raise ValueError(f"Nt = {self.Nt} must be divisible by n = {self.n}")
        part_size = self.Nt // self.n
        bounds = [(i * part_size, (i + 1) * part_size - 1) for i in range(self.n)]
        return bounds

    def find_part(self, t, bounds):
        """Given a coordinate t, find which part it belongs to."""
        for i, (start, end) in enumerate(bounds):
            if start <= t <= end:
                return i, start, end
        raise ValueError(f"Invalid coordinate t = {t}")

    def move(self, txyz, direction, forward: bool, xcutoff):
        """Moves coordinate txyz -> txyz with correct BCs, forward or backward"""
        t, x, y, z = txyz
        if not (0 <= t < self.Nt and 0 <= x < self.Nx and 0 <= y < self.Ny and 0 <= z < self.Nz):
            raise ValueError(f"Invalid coordinates txyz = {txyz}")
        # Initialize all coordinates to their original values
        t_wrapped = t
        x_wrapped = x
        y_wrapped = y
        z_wrapped = z
        if direction == 0:
            if self.is_a(txyz, xcutoff):
                t_wrapped = (t + 1) % self.Nt if forward else (t - 1 + self.Nt) % self.Nt
            else:
                bounds = self.get_part_bounds()
                part_idx, start, end = self.find_part(t, bounds)
                if forward:  
                    t_wrapped = start if t == end else t + 1
                else:
                    t_wrapped = end if t == start else t - 1
        elif direction == 1:
            x_wrapped = (x + 1) % self.Nx if forward else (x - 1 + self.Nx) % self.Nx
        elif direction == 2:
            y_wrapped = (y + 1) % self.Ny if forward else (y - 1 + self.Ny) % self.Ny
        elif direction == 3:
            z_wrapped = (z + 1) % self.Nz if forward else (z - 1 + self.Nz) % self.Nz
        else:
            raise ValueError(f"Invalid direction {direction}, must be 0-3.")
        return [t_wrapped, x_wrapped, y_wrapped, z_wrapped]

    def replica_line_move_forward(self, line, txyz, direction, xcutoff):
        #print("line move forward", txyz, direction)
        link = self.U1[txyz[0], txyz[1], txyz[2], txyz[3], direction, :, :]
        new_txyz = self.move(txyz, direction, True, xcutoff)
        new_line = np.dot(line, link)
        return new_line, new_txyz

    def replica_line_move_backward(self, line, txyz, direction, xcutoff):
        #print("line move backward", txyz, direction)
        new_txyz = self.move(txyz, direction, False, xcutoff)
        link = self.U1[new_txyz[0], new_txyz[1], new_txyz[2], new_txyz[3], direction, :, :].conj().T
        new_line = np.dot(line, link)
        return new_line, new_txyz

    def dS_staple_replica(self, txyz, mu, xcutoff):
        t, x, y, z = txyz
        tmp = np.zeros((3, 3), dtype='complex128')
        for nu in range(4):
            if nu != mu:
                start_txyz = [t, x, y, z]
                start_txyz = self.move(start_txyz, mu, True, xcutoff) # move mu forward
                
                line1 = np.eye(3, dtype='complex128')
                line1, next_txyz = self.replica_line_move_forward(line1, start_txyz, nu, xcutoff)
                line1, next_txyz = self.replica_line_move_backward(line1, next_txyz, mu, xcutoff)
                line1, next_txyz = self.replica_line_move_backward(line1, next_txyz, nu, xcutoff)
                tmp += line1
                
                line2 = np.eye(3, dtype='complex128')
                line2, next_txyz = self.replica_line_move_backward(line2, start_txyz, nu, xcutoff)
                line2, next_txyz = self.replica_line_move_backward(line2, next_txyz, mu, xcutoff)
                line2, next_txyz = self.replica_line_move_forward(line2, next_txyz, nu, xcutoff)
                tmp += line2
        
        return tmp / self.u0**3


    def plaquette_replica(self, txyz, mu, nu, xcutoff):
        result = np.eye(3, dtype='complex128')
        result, next_txyz = self.replica_line_move_forward(result, txyz, mu, xcutoff)
        result, next_txyz = self.replica_line_move_forward(result, next_txyz, nu, xcutoff)
        result, next_txyz = self.replica_line_move_backward(result, next_txyz, mu, xcutoff)
        result, next_txyz = self.replica_line_move_backward(result, next_txyz, nu, xcutoff)
        return result

    def eval_point_S_replica(self, txyz, xcutoff):
        tmp = 0.
        for mu in range(1, 4):  #sum over \mu > \nu spacetime dimensions # CHANGED TO INCLUDE TIMELIKE
            for nu in range(mu):
                tmp += ( 1. - np.real(np.trace( self.plaquette_replica(txyz, mu, nu, xcutoff) )) / 3. / self.u0**4 )
        return self.beta * tmp

    def calc_action_replica(self, xcutoff):
        S = 0.
        for t in range(self.Nt):
            for x in range(self.Nx):
                for y in range(self.Ny):
                    for z in range(self.Nz):
                        txyz = [t, x, y, z]
                        S += self.eval_point_S_replica(txyz, xcutoff)
        return S

    def calc_S_int(self, alpha, xcutoff_1, xcutoff_2):
        S_1 = self.calc_action_replica(xcutoff_1)
        S_2 = self.calc_action_replica(xcutoff_2)
        S_int = (1. - alpha) * S_1 + alpha * S_2
        return S_int

    def deltaS(self, link, updated_link, staple):
        return (-self.beta / 3. / self.u0 ) * np.real(np.trace(np.dot( (updated_link - link), staple)))

    def delta_S_int(self, alpha, dS_1, dS_2):
        #S_1 = S_L and S_2 = S_{L+1}
        dS_int = (1. - alpha) * dS_1 + alpha * dS_2
        return dS_int
    
    def markov_chain_sweep_replica(self, Ncfg, matrices, Nhits, dir_name, alpha, Nstart):
        ratio_accept = 0.
        cfg_cntr = 0
        matrices_length = len(matrices)
        if (self.Nt % self.n) != 0:
            raise ValueError("Nt must be divisible by n for replica lattice.")
        s_delta_s = self.s + 1
        xcutoff_1 = self.Nx - self.s
        xcutoff_2 = self.Nx - s_delta_s
        txyz = np.zeros(4, dtype=int)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        for config in range(Nstart, Nstart + Ncfg - 1):
            #print(f"starting sweep {config}: {datetime.datetime.now()}")
            ### check if config exists and if so, move to next
            filename_U1 = os.path.join(dir_name, f"config_{config:01d}_U1.npy")
            if not os.path.exists(filename_U1):
                cfg_cntr += 1
                for t in range(self.Nt):
                    for x in range(self.Nx):
                        for y in range(self.Ny):
                            for z in range(self.Nz):
                                for mu in range(4):
                                    txyz = [t, x, y, z]
                                    A1 = self.dS_staple_replica(txyz, mu, xcutoff_1)
                                    A2 = self.dS_staple_replica(txyz, mu, xcutoff_2)
                                    for _ in range(Nhits):
                                        r = np.random.randint(0, matrices_length)
                                        matrix = matrices[r]
                                        U = self.U1[t, x, y, z, mu, :, :]
                                        U_prime = np.dot(matrix, U)
                                        dS_1 = self.deltaS(U, U_prime, A1)
                                        dS_2 = self.deltaS(U, U_prime, A2)
                                        dS_int = self.delta_S_int(alpha, dS_1, dS_2)
                                        if (np.exp(-1. * dS_int) > np.random.uniform(0, 1)):
                                            self.U1[t, x, y, z, mu, :, :] = U_prime
                                            ratio_accept += 1
                # Save the configuration
                np.save(filename_U1, self.U1)
        if cfg_cntr > 0:
            ratio_accept = float(ratio_accept) / cfg_cntr / self.Nx / self.Ny / self.Nz / self.Nt / 4. / Nhits
        else:
            ratio_accept = 0.
        print(f"Acceptance ratio for alpha {alpha}: {ratio_accept}")
