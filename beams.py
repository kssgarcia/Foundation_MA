import numpy as np
import solidspy.preprocesor as pre

# def beam(L=10, H=10, E=206.8e9, v=0.28, nx=20, ny=20, dirs=np.array([]), positions=np.array([]), n=1):
#     """
#     This function selects the appropriate beam function to call based on the value of n.

#     Parameters
#     ----------
#     L : float, optional
#         Beam's length, by default 10
#     H : float, optional
#         Beam's height, by default 10
#     F : float, optional
#         Vertical force, by default -1000000
#     E : float, optional
#         Young's modulus, by default 206.8e9
#     v : float, optional
#         Poisson's ratio, by default 0.28
#     nx : int, optional
#         Number of elements in the x direction, by default 20
#     ny : int, optional
#         Number of elements in the y direction, by default 20
#     n : int, optional
#         Selector for the beam function to call, by default 1

#     Returns
#     -------
#     ndarray
#         Nodes array returned by the selected beam function
#     """
#     match n:
#         case 1:
#             return beam_1(L, H, E, v, nx, ny, dirs, positions)
#         case 2:
#             return beam_2(L, H, E, v, nx, ny, dirs, positions)


# def beam_1(L=10, H=10, E=206.8e9, v=0.28, nx=20, ny=20, dirs=np.array([]), positions=np.array([])):
#     """
#     Make the mesh for a cuadrilateral model with cantilever beam's constrains.

#     Parameters
#     ----------
#     L : float, optional
#         Length of the beam, by default 10
#     H : float, optional
#         Height of the beam, by default 10
#     E : float, optional
#         Young's modulus, by default 206.8e9
#     v : float, optional
#         Poisson's ratio, by default 0.28
#     nx : int, optional
#         Number of elements in the x direction, by default 20
#     ny : int, optional
#         Number of elements in the y direction, by default 20
#     dirs : ndarray, optional
#         An array with the directions of the loads, by default empty array. [[0,1],[1,0],[0,-1]]
#     positions : ndarray, optional
#         An array with the positions of the loads, by default empty array. [[61,30], [1,30], [30, 1]]

#     Returns
#     -------
#     nodes : ndarray
#         Array of nodes
#     mats : ndarray
#         Array of material properties
#     els : ndarray
#         Array of elements
#     loads : ndarray
#         Array of loads
#     """
#     x, y, els = pre.rect_grid(L, H, nx, ny)
#     mats = np.zeros((els.shape[0], 3))
#     mats[:] = [E,v,1]
#     nodes = np.zeros(((nx + 1)*(ny + 1), 5))
#     nodes[:, 0] = range((nx + 1)*(ny + 1))
#     nodes[:, 1] = x
#     nodes[:, 2] = y
#     mask = (x==-L/2)
#     nodes[mask, 3:] = -1

#     loads = np.zeros((dirs.shape[0], 3), dtype=int)
#     node_index = nx*positions[:,0]+(positions[:,0]-positions[:,1])

#     loads[:, 0] = node_index
#     loads[:, 1] = dirs[:,0]
#     loads[:, 2] = dirs[:,1]
#     BC = nodes[mask, 0]
#     return nodes, mats, els, loads, BC

# def beam_2(L=10, H=10, E=206.8e9, v=0.28, nx=20, ny=20, dirs=np.array([]), positions=np.array([])):
#     """
#     Make the mesh for a cuadrilateral model with simply supported beam's constrains.

#     Parameters
#     ----------
#     L : float, optional
#         Length of the beam, by default 10
#     H : float, optional
#         Height of the beam, by default 10
#     E : float, optional
#         Young's modulus, by default 206.8e9
#     v : float, optional
#         Poisson's ratio, by default 0.28
#     nx : int, optional
#         Number of elements in the x direction, by default 20
#     ny : int, optional
#         Number of elements in the y direction, by default 20
#     dirs : ndarray, optional
#         An array with the directions of the loads, by default empty array. [[0,1],[1,0],[0,-1]]
#     positions : ndarray, optional
#         An array with the positions of the loads, by default empty array. [[61,30], [1,30], [30, 1]]

#     Returns
#     -------
#     nodes : ndarray
#         Array of nodes
#     mats : ndarray
#         Array of material properties
#     els : ndarray
#         Array of elements
#     loads : ndarray
#         Array of loads
#     """
#     x, y, els = pre.rect_grid(L, H, nx, ny)
#     mats = np.zeros((els.shape[0], 3))
#     mats[:] = [E,v,1]
#     nodes = np.zeros(((nx + 1)*(ny + 1), 5))
#     nodes[:, 0] = range((nx + 1)*(ny + 1))
#     nodes[:, 1] = x
#     nodes[:, 2] = y
#     mask_1 = (x == L/2) & (y > H/4)
#     mask_2 = (x == L/2) & (y < -H/4)
#     mask = np.bitwise_or(mask_1, mask_2)
#     nodes[mask, 3:] = -1

#     loads = np.zeros((dirs.shape[0], 3), dtype=int)
#     node_index = nx*positions[:,0]+(positions[:,0]-positions[:,1])

#     loads[:, 0] = node_index
#     loads[:, 1] = dirs[:,0]
#     loads[:, 2] = dirs[:,1]
#     BC = nodes[mask, 0]
#     return nodes, mats, els, loads, BC

def beam(L=10, H=10, E=206.8e9, v=0.28, nx=20, ny=20, dirs=np.array([]), positions=np.array([])):
    """
    Make the mesh for a cuadrilateral model with cantilever beam's contrains.

    Parameters
    ----------
    L : float (optional)
        Beam's lenght
    H : float (optional)
        Beam's height
    F : float (optional)
        Vertical force.
    E : string (optional)
        Young module
    v : string (optional)
        Poisson ratio
    nx : int (optional)
        Number of element in x direction
    ny : int (optional)
        Number of element in y direction

    Returns
    -------
    nodes : ndarray
        Nodes array
    mats : ndarray (1, 2)
        Mats array
    els : ndarray
        Elements array
    loads : ndarray
        Loads array
    BC : ndarray
        Boundary conditions nodes

    """
    x, y, els = pre.rect_grid(L, H, nx, ny)
    mats = np.zeros((els.shape[0], 3))
    mats[:] = [E,v,1]
    nodes = np.zeros(((nx + 1)*(ny + 1), 5))
    nodes[:, 0] = range((nx + 1)*(ny + 1))
    nodes[:, 1] = x
    nodes[:, 2] = y
    nodes[(x==-L/2), 3] = -1
    nodes[(x==L/2), 3] = -1
    nodes[(y==-H/2), 3] = -1

    selected_nodes = nodes[(x==0)&(y==H/2), 0]
    loads = np.zeros((selected_nodes.shape[0], 3), dtype=int)

    loads[:, 0] = selected_nodes
    loads[:, 1] = dirs[:,0]
    loads[:, 2] = dirs[:,1]
    print(loads)
    
    return nodes, mats, els, loads