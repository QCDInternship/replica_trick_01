import os, sys, string
import numpy as np
import gauge_latticeqcd as gl


action = 'W'
Nt, Nx, Ny, Nz = 6, 6, 6, 6
beta = 5.7
run_number = 1

alpha_to_check = 0.0
ensemble_to_check = 0


Nstart = 0
Nend = 3000 
base_dir = f"RunReplica{run_number}"
dir_name = f"Replica_{action}_{Nt}x{Nx}x{Ny}x{Nz}_b{int(beta * 100)}_alpha{alpha_to_check}_ensemble{ensemble_to_check}"
full_path = os.path.join(base_dir, dir_name)

# Prepare output file
outfile = f'./plaquette_thermalization_{int(beta * 100)}_{Nt}x{Nx}_{run_number}.dat'
fout = open(outfile, 'w') # Use 'w' to overwrite old analysis files

fout.write('#1:cfg  2:plaquette\n')

print(f"Analyzing configurations from: {full_path}")
print(f"Writing output to: {outfile}")

# Loop over the generated configurations
for Ncfg in range(Nstart, Nend + 1):
    # Construct the specific filename for each configuration
    infile = os.path.join(full_path, f"config_{Ncfg}_U1.npy")
    
    # Check if the file exists before trying to load it
    if os.path.exists(infile):
        # Load the lattice configuration
        U = np.load(infile)
        
        # Calculate the average plaquette value
        pl = gl.fn_average_plaquette(U)
        
        # Write the result to the output file
        fout.write(str(Ncfg) + ' ' + str(pl) + '\n')
    else:
        # If a file is missing in the sequence, print a warning
        print(f"Warning: Configuration file not found, skipping: {infile}")

fout.close()
print("Analysis complete.")