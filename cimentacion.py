# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import solidspy.postprocesor as pos     # Rutinas de postprocesado
import aux_functions as aux             # Rutinas externas al programa y creadas para la automatización de algunas tareas.
from utils import *
from beams import *

# Initialize variables
length = 60
height = 60
nx = 60
ny= 60
niter = 60
penal = 3 # Penalization factor
Emin=1e-9 # Minimum young modulus of the material
Emax=1.0 # Maximum young modulus of the material
n = nx*ny

dirs = np.array([[0,-1]])
positions = np.array([[61,30]])
nodes, mats, els, loads = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions)

# Initialize the design variables
change = 10 # Change in the design variable
g = 0 # Constraint
rho = 0.5 * np.ones(ny*nx, dtype=float) # Initialize the density
sensi_rho = np.ones(ny*nx) # Initialize the sensitivity
rho_old = rho.copy() # Initialize the density history
d_c = np.ones(ny*nx) # Initialize the design change

r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4 # Radius for the sensitivity filter
centers = center_els(nodes, els) # Calculate centers
E = mats[0,0] # Young modulus
nu = mats[0,1] # Poisson ratio

for _ in range(1):

    # Check convergence
    if change < 0.01:
        print('Convergence reached')
        break

    # Change density 
    mats[:,2] = Emin+rho**penal*(Emax-Emin)

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
    print(change)

tri = Triangulation(nodes[:,1], nodes[:,2], els[:, -3:])

plt.figure(figsize=(8, 6))
plt.tripcolor(tri, rho, edgecolors='k', cmap='gray')  # Using grayscale colormap
plt.colorbar(label='Color')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mesh Elements with Grayscale Colors')
plt.grid(True)
plt.show()

# %%

aux.deformacionFill(nodes , els , UC , factor = 1.0 , cmap='cividis')

# %%

pos.fields_plot(elements, nodes, UC , S_nodes=S_nodes)