# %%
import time
import numpy as np
from scipy.sparse.linalg import spsolve
import solidspy.assemutil as ass # Solidspy 1.1.0
import solidspy.postprocesor as pos 
import aux_functions as aux             # Rutinas externas al programa y creadas para la automatización de algunas tareas.

import matplotlib.pyplot as plt 
from matplotlib import colors

# from utils.beams import beam, beamNormal
from utils.beams import beamSimp
from utils.SIMP_utils_test import sparse_assem, optimality_criteria, density_filter, center_els, sensi_el

# Start the timer
start_time = time.time()

np.seterr(divide='ignore', invalid='ignore')

n_elem = 60

# Initialize variables
length = 60
height = 60
nx = n_elem
ny= n_elem
niter = 60
penal = 3 # Penalization factor
Emin=1. # Minimum young modulus of the material
Emax=10. # Maximum young modulus of the material
poisson_max = 0.3
poisson_min = 0.3

dirs = np.array([[0,-1]])
positions = np.array([[61,30]])
nodes, mats, els, loads, found_nodes = beamSimp(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions)

# Initialize the design variables
change = 10 # Change in the design variable
g = 0 # Constraint
sensi_number = 0.5*np.ones(ny*nx) # Initialize the sensitivity
sensi_number_old = sensi_number.copy() 
d_c = np.ones(ny*nx) # Initialize the design change

assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8) 

mats[:, 0] = Emin  # Update Young's modulus
mats[:, 1] = poisson_min  # Update Poisson's ratio
print(mats)

plt.ion() 
fig,ax = plt.subplots()
ax.imshow(np.flipud(-mats[:,0].reshape(nx,ny)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax.set_title('Mats density initial')
fig.show()

plt.ion() 
fig,ax = plt.subplots()
ax.imshow(np.flipud(-sensi_number.reshape(nx,ny)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax.set_title('Rho initial')
fig.show()

iter = 0
for _ in range(niter):
    iter += 1

    # Check convergence
    if change < 0.01:
        print('Convergence reached')
        break

    # Change density 
    mats[:,2] = Emin+sensi_number**penal*(Emax-Emin)

    # FEM
    stiff_mat = sparse_assem(els, mats, neq, assem_op, nodes)
    rhs_vec = ass.loadasem(loads, bc_array, neq)
    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp)

    # Sensitivity analysis
    sensi_number[:] = sensi_el(nodes, mats, els, UC)
    sensi_number = sensi_number/np.abs(sensi_number).max()

    # Update material properties
    mask_foundation = sensi_number >= 0.5
    mask_soil = sensi_number < 0.5

    mats[mask_foundation, 0] = Emax  # Update Young's modulus
    mats[mask_foundation, 1] = poisson_max  # Update Poisson's ratio
    mats[mask_soil, 0] = Emin  # Update Young's modulus
    mats[mask_soil, 1] = poisson_min  # Update Poisson's ratio

    # Compute the change
    change = np.linalg.norm(sensi_number.reshape(nx*ny,1)-sensi_number_old.reshape(nx*ny,1),np.inf)
    print(change)

# aux.deformacionFill(nodes , els , UC , factor = 1.0 , cmap='cividis')

plt.ion() 
fig,ax = plt.subplots()
ax.imshow(np.flipud(-sensi_number.reshape(nx,ny)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax.set_title('Rho')
fig.show()