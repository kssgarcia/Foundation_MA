# %%
import time
import numpy as np
from scipy.sparse.linalg import spsolve
import solidspy.postprocesor as pos 
import solidspy.assemutil as ass # Solidspy 1.1.0
import aux_functions as aux    

import matplotlib.pyplot as plt 
from matplotlib import colors

from utils.beams import beam 
from utils.SIMP_utils import sparse_assem, optimality_criteria, density_filter, center_els, sensi_el

# Start the timer
start_time = time.time()

np.seterr(divide='ignore', invalid='ignore')

def optimization(n_elem):
    # Initialize variables
    length = 60
    height = 60
    nx = n_elem
    ny= n_elem
    penal = 3 
    # Emin=1e-9 # Minimum young modulus of the material
    # Emax=1.0 # Maximum young modulus of the material
    Emax=2e9 
    Emin=0.0049 
    poisson_max = 0.15
    poisson_min = 0.28

    dirs = np.array([[0,-1]])
    positions = np.array([[61,30]])
    nodes, mats, els, loads, found_nodes = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions)

    # NODES, ELS = nodes, els.copy()

    # Design variables initialization
    change = 10 # Change in the design variable
    g = 0 # Constraint
    rho = 0.4 * np.ones(ny*nx, dtype=float) # Initialize the density
    sensi_rho = np.ones(ny*nx) # Initialize the sensitivity
    rho_old = rho.copy() # Initialize the density history
    d_c = np.ones(ny*nx) # Initialize the design change
    r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4 # Radius for the sensitivity filter
    centers = center_els(nodes, els) # Calculate centers
    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8) 

    # Foundation initialization
    found_elements = [i for i, el in enumerate(els[:,-4:]) if any(node in el for node in found_nodes)]
    rho[found_elements] = 1

    # Update material properties
    mask_foundation = rho >= 0.95
    mask_soil = rho < 0.95

    mats[mask_foundation, 0] = Emax  # Update Young's modulus
    mats[mask_foundation, 1] = poisson_max  # Update Poisson's ratio
    mats[mask_soil, 0] = Emin  # Update Young's modulus
    mats[mask_soil, 1] = poisson_min  # Update Poisson's ratio

    plt.ion() 
    fig,ax = plt.subplots()
    ax.imshow(np.flipud(-mats[:,0].reshape(nx,ny)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    ax.set_title('Mats density initial')
    fig.show()

    plt.ion() 
    fig,ax = plt.subplots()
    ax.imshow(np.flipud(-rho.reshape(nx,ny)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    ax.set_title('Rho initial')
    fig.show()

    iter = 0
    for _ in range(100):
        iter += 1

        # Check convergence
        if change < 0.01:
            print('Convergence reached')
            break

        # Updata mats density
        mats[:,2] = Emin+rho**penal*(Emax-Emin)

        # System assembly
        stiff_mat = sparse_assem(els, mats, neq, assem_op, nodes)
        rhs_vec = ass.loadasem(loads, bc_array, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(bc_array, nodes, disp)
        _, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)

        # Sensitivity analysis
        sensi_rho[:] = sensi_el(nodes, mats, els, UC)
        d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
        d_c[:] = density_filter(centers, r_min, rho, d_c)
        rho_old[:] = rho.copy()
        rho[:], g = optimality_criteria(nx, ny, rho, d_c, g)

        mask_foundation = rho >= 0.95
        mask_soil = rho < 0.95
        print(rho.sum(), rho_old.sum())
        print(np.count_nonzero(mask_foundation))
        print(np.count_nonzero(mask_soil))

        # Update values
        mats[mask_foundation, 0] = Emax  # Update Young's modulus
        mats[mask_foundation, 1] = poisson_max  # Update Poisson's ratio

        mats[mask_soil, 0] = Emin  # Update Young's modulus
        mats[mask_soil, 1] = poisson_min  # Update Poisson's ratio

        # Compute the change
        change = np.linalg.norm(rho.reshape(nx*ny,1)-rho_old.reshape(nx*ny,1),np.inf)

        print(change)
        if iter % 10 == 0:
            print('Stop')
            break

    plt.ion() 
    fig,ax = plt.subplots()
    ax.imshow(np.flipud(-mats[:,0].reshape(nx,ny)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    ax.set_title('Mats density')
    fig.show()

    plt.ion() 
    fig,ax = plt.subplots()
    ax.imshow(np.flipud(-rho.reshape(nx,ny)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    ax.set_title('Rho')
    fig.show()

    # aux.deformacionFill(NODES , ELS , UC , factor = 1.0 , cmap='cividis')
    pos.fields_plot(els, nodes, UC , S_nodes=S_nodes)

if __name__ == "__main__":
    optimization(60)

