# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import solidspy.postprocesor as pos     # Rutinas de postprocesado
import aux_functions as aux             # Rutinas externas al programa y creadas para la automatización de algunas tareas.
import pandas as pd
from utils import *

z     = [2.0 , 5.0 , 8.0 , 12.0 , 22]
phi   = [23.0, 28.0, 29.0 , 30.0 , 30.0]
gamma = [1.6 , 1.7 , 1.85 , 1.80 , 1.90]
Sc = aux.properties(z , phi, gamma)
pd.DataFrame(Sc)

id_model = 1   # (0): Zapata (1): 1 pilote (2): 2 pilotes 
P1 =-2517.0
P2 = 2760.0
V1 = 76.5
V2 = 76.5

eles = aux.zapata( P1 , P2 , V1 , V2)
nodes     = np.loadtxt('files/Pnodes.txt', ndmin=2)
elements  = np.loadtxt('files/Peles.txt', ndmin=2, dtype=int)
loads     = np.loadtxt('files/Ploads.txt', ndmin=2)    
els_obs = np.vstack((eles[3] , eles[4] , eles[5] , eles[6] , eles[7] ))

mats = np.zeros((elements.shape[0], 3))
mats[:] = [6000.00, 0.300000, 1]

change = 10 # Change in the design variable
g = 0 # Constraint
n = elements.shape[0]
rho = 0.5 * np.ones(n, dtype=float) # Initialize the density
sensi_rho = np.ones(n) # Initialize the sensitivity
rho_old = rho.copy() # Initialize the density history
d_c = np.ones(n) # Initialize the design change
penal = 3 # Penalization factor
Emin=1e-9 # Minimum young modulus of the material
Emax=6000.00 # Maximum young modulus of the material
centers = center_els(nodes, elements) # Calculate centers

r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4 # Radius for the sensitivity filter

niter = 60
iter = 0
for _ in range(niter):
    iter += 1

    # Check convergence
    if change < 0.01:
        print('Convergence reached')
        break

    # Change density 
    mats[:,2] = Emin+rho**penal*(Emax-Emin)

    IBC, disp, rhs_vec  = preprocessing(nodes, mats, elements, loads) 
    UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], elements, IBC, disp)

    compliance = rhs_vec.T.dot(disp)

    # Sensitivity analysis
    sensi_rho[:] = sensi_el(nodes, mats, elements, UC)
    d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
    d_c[:] = density_filter(centers, r_min, rho, d_c)

    # Optimality criteria
    rho_old[:] = rho
    rho[:], g = optimality_criteria(n, rho, d_c, g)

    # Compute the change
    change = np.linalg.norm(rho.reshape(n,1)-rho_old.reshape(n,1),np.inf)
    print(change)

tri = Triangulation(nodes[:,1], nodes[:,2], elements[:, -3:])

plt.figure(figsize=(8, 6))
plt.tripcolor(tri, rho, edgecolors='k', cmap='gray')  # Using grayscale colormap
plt.colorbar(label='Color')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mesh Elements with Grayscale Colors')
plt.grid(True)
plt.show()

# %%

aux.deformacionFill(nodes , elements , UC , factor = 1.0 , cmap='cividis')

# %%

pos.fields_plot(elements, nodes, UC , S_nodes=S_nodes)