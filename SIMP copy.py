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
from utils.SIMP_utils_test import sparse_assem, strain_els, sensi_el

# Start the timer
start_time = time.time()

np.seterr(divide='ignore', invalid='ignore')

# VARIABLES INITIALIZATION
length = 60
height = 120
nx = 60
ny = 120
niter = 60
Emin = 1
Emax = 10
poisson_max = 0.3
poisson_min = 0.3

dirs = np.array([[0,-1]])
positions = np.array([[121,61]])
nodes, mats, els, loads, found_nodes = beamSimp(L=nx, H=ny, nx=length, ny=height, dirs=dirs, positions=positions)

change = 10
sensi_number = 0.1*np.ones(ny*nx) 
sensi_number_old = sensi_number.copy() 
assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8) 

# ESO
mats[:, 0] = Emax  # Update Young's modulus
mats[:, 1] = poisson_max  # Update Young's modulus

# BESO
# found_elements = [i for i, el in enumerate(els[:,-4:]) if any(node in el for node in found_nodes)]
# sensi_number[found_elements] = 1
# mask_foundation = sensi_number >= 0.95
# mask_soil = sensi_number < 0.95
# print(np.count_nonzero(mask_foundation))
# print(np.count_nonzero(mask_soil))
# mats[mask_foundation, 0] = Emax  # Update Young's modulus
# mats[mask_foundation, 1] = poisson_max  # Update Poisson's ratio
# mats[mask_soil, 0] = Emin  # Update Young's modulus
# mats[mask_soil, 1] = poisson_min  # Update Poisson's ratio

plt.ion() 
fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.flipud(-mats[:,0].reshape(ny,nx)), cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-Emax,vmax=Emin))
ax[0].set_title('Mats density initial')
ax[1].imshow(np.flipud(-sensi_number.reshape(ny,nx)), cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1,vmax=0))
ax[1].set_title('sensi_number initial')
fig.show()

RR = 0.001 # Initial removal ratio
ER = 0.005 # Removal ratio increment
iter = 0
for iter in range(30):
    print(f'---------------------- Iteration {iter} ----------------------')
    # Check convergence
    if change < 0.01:
        print('Convergence reached')
        break

    # Change density 
    mats[:,2] = sensi_number

    # FEM
    stiff_mat = sparse_assem(els, mats, neq, assem_op, nodes)
    rhs_vec = ass.loadasem(loads, bc_array, neq)
    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp)
    E_nodes, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)
    E_els, S_els = strain_els(els, E_nodes, S_nodes) # Calculate strains and stresses in elements

    # ESO
    vons = np.sqrt(S_els[:,0]**2 - (S_els[:,0]*S_els[:,1]) + S_els[:,1]**2 + 3*S_els[:,2]**2)
    sensi_number = vons/vons.max() # Relative stress
    mask_del = sensi_number < RR # Mask for elements to be deleted
    mats[mask_del, 0] = Emin  # Update Young's modulus
    mats[mask_del, 1] = poisson_min  # Update Poisson's ratio
    print(np.count_nonzero(mask_del))
    RR += ER

    # BESO
    # sensi_number[:] = sensi_el(nodes, mats, els, UC)
    # sensi_number = sensi_number/np.abs(sensi_number).max()
    # mask_foundation = sensi_number >= 0.2
    # mask_soil = sensi_number < 0.2
    # print(np.count_nonzero(mask_foundation))
    # print(np.count_nonzero(mask_soil))
    # mats[mask_foundation, 0] = Emax  # Update Young's modulus
    # mats[mask_foundation, 1] = poisson_max  # Update Poisson's ratio
    # mats[mask_soil, 0] = Emin  # Update Young's modulus
    # mats[mask_soil, 1] = poisson_min  # Update Poisson's ratio

    # Compute the change
    change = np.linalg.norm(sensi_number.reshape(nx*ny,1)-sensi_number_old.reshape(nx*ny,1),np.inf)
    print(change)


    plt.ion() 
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.flipud(-mats[:,0].reshape(ny,nx)), cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-Emax,vmax=Emin))
    ax[0].set_title(f'Mats iter:{iter}')
    ax[1].imshow(np.flipud(-sensi_number.reshape(ny,nx)), cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1,vmax=0))
    ax[1].set_title(f'sensi_number iter:{iter}')
    fig.show()


aux.deformacionFill(nodes , els , UC , factor = 1.0 , cmap='cividis')

a = np.arange(sensi_number.shape[0])
sensi_number_list = sensi_number.tolist()
plt.plot(a.tolist(), sensi_number_list)
# Add title and labels
plt.title('Distribution of sensi_number')
plt.xlabel('Value')
plt.ylabel('Frequency')
# Show plot
plt.show()
