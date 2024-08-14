# %% Imports
import matplotlib.pyplot as plt # Package for plotting
import numpy as np # Package for scientific computing

from utils.beams import beamBeso # Functions for mesh generation
from utils.BESO_utils import is_equilibrium, preprocessing, postprocessing, protect_els, del_node, volume, sensitivity_els, adjacency_nodes, center_els, sensitivity_nodes, sensitivity_filter
# Solidspy 1.1.0
import solidspy.postprocesor as pos # SolidsPy package for postprocessing
np.seterr(divide='ignore', invalid='ignore') # Ignore division by zero error

length = 60
height = 60
nx = 60
ny= 60
niter = 20
ER = 0.01 # Removal ratio increment
t = 0.0001 # Threshold for error

dirs = np.array([[0,-1]])
positions = np.array([[61,30]])
nodes, mats, els, loads, found_nodes, BC = beamBeso(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions)

elsI, nodesI = np.copy(els), np.copy(nodes) # Copy mesh
IBC, UG, _ = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 1 # Radius for the sensitivity filter
adj_nodes = adjacency_nodes(nodes, els) # Adjacency nodes
centers = center_els(nodes, els) # Centers of elements

Vi = volume(els, length, height, nx, ny) # Initial volume
V_opt = Vi.sum() * 0.50 # Optimal volume

# Initialize variables.
ELS = None
mask = np.ones(els.shape[0], dtype=bool) # Mask of elements to be removed
# found_elements = [i for i, el in enumerate(els[:,-4:]) if any(node in el for node in found_nodes)]
# mask[found_elements] = True
sensi_I = None  
C_h = np.zeros(niter) # History of compliance
error = 1000 

for i in range(niter):
    # Calculate the optimal design array elements
    els_del = els[mask].copy() # Elements to be removed
    V = Vi[mask].sum() # Volume of the structure

    # Check equilibrium
    if not is_equilibrium(nodes, mats, els_del, loads):  
        print('Is not equilibrium')
        break # Stop the program if the structure is not in equilibrium

    # Storage the solution
    ELS = els_del 

    # FEW analysis
    IBC, UG, rhs_vec = preprocessing(nodes, mats, els_del, loads) # Calculate boundary conditions and global stiffness matrix
    UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els_del, IBC, UG) # Calculate displacements, strains and stresses

    # Sensitivity filter
    sensi_e = sensitivity_els(nodes, mats, els, mask, UC) # Calculate the sensitivity of the elements
    sensi_nodes = sensitivity_nodes(nodes, adj_nodes, centers, sensi_e) # Calculate the sensitivity of the nodes
    sensi_number = sensitivity_filter(nodes, centers, sensi_nodes, r_min) # Perform the sensitivity filter

    # Average the sensitivity numbers to the historical information 
    if i > 0: 
        sensi_number = (sensi_number + sensi_I)/2 # Average the sensitivity numbers to the historical information
    sensi_number = sensi_number/sensi_number.max() # Normalize the sensitivity numbers

    # Check if the optimal volume is reached and calculate the next volume
    V_r = False
    if V <= V_opt:
        els_k = els_del.shape[0]
        V_r = True
        break
    else:
        V_k = V * (1 + ER) if V < V_opt else V * (1 - ER)

    # Remove/add threshold
    sensi_sort = np.sort(sensi_number)[::-1] # Sort the sensitivity numbers
    els_k = els_del.shape[0]*V_k/V # Number of elements to be removed
    alpha_del = sensi_sort[int(els_k)] # Threshold for removing elements

    # Remove/add elements
    mask = sensi_number > alpha_del # Mask of elements to be removed
    mask_els = protect_els(els[np.invert(mask)], els.shape[0], loads, BC) # Mask of elements to be protected
    mask = np.bitwise_or(mask, mask_els) 
    del_node(nodes, els[mask], loads, BC) # Delete nodes

    # Calculate the strain energy and storage it 
    C = 0.5*rhs_vec.T@UG
    C_h[i] = C
    if i > 10: error = C_h[i-5:].sum() - C_h[i-10:-5].sum()/C_h[i-5:].sum()

    # Check for convergence
    if error <= t and V_r == True:
        print("convergence")
        break

    # Save the sensitvity number for the next iteration
    sensi_I = sensi_number.copy()
# %%

pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI) 
pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes) 

fill_plot = np.ones(E_nodes.shape[0])
plt.figure()
tri = pos.mesh2tri(nodes, ELS)
plt.tricontourf(tri, fill_plot, cmap='binary')
plt.axis("image");

fill_plot = np.ones(E_nodes.shape[0])
plt.figure()
tri = pos.mesh2tri(nodes, elsI)
plt.tricontourf(tri, fill_plot, cmap='binary')
plt.axis("image");