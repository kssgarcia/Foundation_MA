# %%
import time
import random
import numpy as np
from scipy.sparse.linalg import spsolve
import solidspy.assemutil as ass # Solidspy 1.1.0
import aux_functions as aux             # Rutinas externas al programa y creadas para la automatización de algunas tareas.

import matplotlib.pyplot as plt 
from matplotlib import colors

from beams import *
from SIMP_utils import *

# Start the timer
start_time = time.time()

np.seterr(divide='ignore', invalid='ignore')


def optimization(n_elem):
    # Initialize variables
    niter = 60
    penal = 3 # Penalization factor
    Emin=1e-9 # Minimum young modulus of the material
    Emax=1.0 # Maximum young modulus of the material

    id_model = 1   # (0): Zapata (1): 1 pilote (2): 2 pilotes 
    P1 =-2517.0
    P2 = 2760.0
    V1 = 76.5
    V2 = 76.5

    eles = aux.zapata( P1 , P2 , V1 , V2)
    nodes     = np.loadtxt('files/Pnodes.txt', ndmin=2)
    els  = np.loadtxt('files/Peles.txt', ndmin=2, dtype=int)
    loads     = np.loadtxt('files/Ploads.txt', ndmin=2)    
    els_obs = np.vstack((eles[3] , eles[4] , eles[5] , eles[6] , eles[7] ))

    mats = np.zeros((els.shape[0], 3))
    mats[:] = [6000.00, 0.300000, 1]
    n = els.shape[0]




    # Initialize the design variables
    change = 10 # Change in the design variable
    g = 0 # Constraint
    rho = 0.5 * np.ones(n, dtype=float) # Initialize the density
    sensi_rho = np.ones(n) # Initialize the sensitivity
    rho_old = rho.copy() # Initialize the density history
    d_c = np.ones(n) # Initialize the design change

    r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4 # Radius for the sensitivity filter
    centers = center_cimen(nodes, els) # Calculate centers
    E = mats[0,0] # Young modulus
    nu = mats[0,1] # Poisson ratio

    iter = 0
    for _ in range(niter):
        iter += 1

        # Check convergence
        if change < 0.01:
            print('Convergence reached')
            break

        # Change density 
        mats[:,2] = Emin+rho**penal*(Emax-Emin)

        # System assembly
        IBC, disp, rhs_vec  = preprocessing(nodes, mats, els, loads) 
        UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els, IBC, disp)

        compliance = rhs_vec.T.dot(disp)

        # Sensitivity analysis
        sensi_rho[:] = sensi_el(nodes, mats, els, UC)
        d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
        d_c[:] = density_filter(centers, r_min, rho, d_c)

        # Optimality criteria
        rho_old[:] = rho
        rho[:], g = optimality_criteria(n, rho, d_c, g)

        # Compute the change
        change = np.linalg.norm(rho.reshape(n,1)-rho_old.reshape(n,1),np.inf)


    from matplotlib.tri import Triangulation
    tri = Triangulation(nodes[:,1], nodes[:,2], els[:, -3:])

    plt.figure(figsize=(8, 6))
    plt.tripcolor(tri, sensi_rho, edgecolors='k', cmap='gray')  # Using grayscale colormap
    plt.colorbar(label='Color')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mesh Elements with Grayscale Colors')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    optimization(60)
