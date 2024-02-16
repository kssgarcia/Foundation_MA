import numpy as np
import solidspy.preprocesor as pre


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
    mask = (x==-L/2)
    nodes[mask, 3:] = -1

    loads = np.zeros((dirs.shape[0], 3), dtype=int)
    node_index = nx*positions[:,0]+(positions[:,0]-positions[:,1])

    loads[:, 0] = node_index
    loads[:, 1] = dirs[:,0]
    loads[:, 2] = dirs[:,1]
    
    return nodes, mats, els, loads