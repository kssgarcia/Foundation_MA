import numpy as np
import solidspy.uelutil as uel 
from scipy.sparse import coo_matrix

from scipy.spatial.distance import cdist

def sparse_assem(elements, mats, neq, assem_op, nodes):
    """
    Assembles the global stiffness matrix
    using a sparse storing scheme

    Parameters
    ----------
    elements : ndarray (int)
      Array with the number for the nodes in each element.
    mats    : ndarray (float)
      Array with the material profiles.
    neq : int
      Number of active equations in the system.
    assem_op : ndarray (int)
      Assembly operator.
    uel : callable function (optional)
      Python function that returns the local stiffness matrix.
    kloc : ndarray 
      Stiffness matrix of a single element

    Returns
    -------
    stiff : sparse matrix (float)
      Array with the global stiffness matrix in a sparse
      Compressed Sparse Row (CSR) format.
    """
    rows = []
    cols = []
    stiff_vals = []
    nels = elements.shape[0]
    for ele in range(nels):
        params = tuple(mats[elements[ele, 0], :])
        elcoor = nodes[elements[ele, -4:], 1:3]
        kloc, _ = uel.elast_quad4(elcoor, params)
        kloc_ = kloc * mats[elements[ele, 0], 2]
        ndof = kloc.shape[0]
        dme = assem_op[ele, :ndof]
        for row in range(ndof):
            glob_row = dme[row]
            if glob_row != -1:
                for col in range(ndof):
                    glob_col = dme[col]
                    if glob_col != -1:
                        rows.append(glob_row)
                        cols.append(glob_col)
                        stiff_vals.append(kloc_[row, col])

    stiff = coo_matrix((stiff_vals, (rows, cols)), shape=(neq, neq)).tocsr()

    return stiff
    
def optimality_criteria(nelx, nely, rho, d_c, g):
    """
    Optimality criteria method.

    Parameters
    ----------
    nelx : int
        Number of elements in x direction.
    nely : int
        Number of elements in y direction.
    rho : ndarray
        Array with the density of each element.
    d_c : ndarray
        Array with the derivative of the compliance.
    g : float
        Volume constraint.

    Returns
    -------
    rho_new : ndarray
        Array with the new density of each element.
    gt : float
        Volume constraint.
    """
    l1=0
    l2=1e9
    move=0.2
    d_v = np.ones(nely*nelx)
    rho_new=np.zeros(nelx*nely)
    while (l2-l1)/(l1+l2)>1e-3: 
        lmid=0.5*(l2+l1)
        rho_new[:]= np.maximum(0.0,np.maximum(rho-move,np.minimum(1.0,np.minimum(rho+move,rho*np.sqrt(-d_c/d_v/lmid)))))
        gt=g+np.sum((rho_new-rho))
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return (rho_new,gt)




def volume(els, length, height, nx, ny):
    """
    Volume calculation.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements.
    length : ndarray
        Length of the beam.
    height : ndarray
        Height of the beam.
    nx : float
        Number of elements in x direction.
    ny : float
        Number of elements in y direction.

    Return 
    ----------
    V: float
        Volume of a single element.
    """

    dy = length / nx
    dx = height / ny
    V = dx * dy * np.ones(els.shape[0])

    return V

def density_filter(centers, r_min, rho, d_rho):
    """
    Performe the sensitivity filter.
    
    Parameters
    ----------
    centers : ndarray
        Array with the centers of each element.
    r_min : float
        Minimum radius of the filter.
    rho : ndarray
        Array with the density of each element.
    d_rho : ndarray
        Array with the derivative of the density of each element.
        
    Returns
    -------
    densi_els : ndarray
        Sensitivity of each element with filter
    """
    dist = cdist(centers, centers, 'euclidean')
    delta = r_min - dist
    H = np.maximum(0.0, delta)
    densi_els = (rho*H*d_rho).sum(1)/(H.sum(1)*np.maximum(0.001,rho))

    return densi_els

def center_els(nodes, els):
    """
    Calculate the center of each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes.
    els : ndarray
        Array with models elements.
        
    Returns
    -------
    centers : 
        Centers of each element.
    """
    centers = np.zeros((els.shape[0], 2))
    for el in els:
        n = nodes[el[-4:], 1:3]
        center = np.array([n[1,0] + (n[0,0] - n[1,0])/2, n[2,1] + (n[0,1] - n[2,1])/2])
        centers[int(el[0])] = center

    return centers

def sensi_el(nodes, mats, els, UC):
    """
    Calculate the sensitivity number for each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    UC : ndarray
        Displacements at nodes

    Returns
    -------
    sensi_number : ndarray
        Sensitivity number for each element.
    """   

    sensi_number = []
    for el in range(len(els)):
        params = tuple(mats[els[el, 0], :])
        elcoor = nodes[els[el, -4:], 1:3]
        kloc, _ = uel.elast_quad4(elcoor, params)
        node_el = els[el, -4:]
        U_el = UC[node_el]
        U_el = np.reshape(U_el, (8,1))
        a_i = U_el.T.dot(kloc.dot(U_el))[0,0]
        sensi_number.append(a_i)
    sensi_number = np.array(sensi_number)

    return sensi_number

def sensitivity_els(nodes, mats, els, UC, nx, ny):
    """
    Calculate the sensitivity number for each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    UC : ndarray
        Displacements at nodes
    nx : float
        Number of elements in x direction.
    ny : float
        Number of elements in y direction.

    Returns
    -------
    sensi_number : ndarray
        Sensitivity number for each element.
    """   
    sensi_number = np.zeros(els.shape[0])
    params = tuple(mats[1, :])
    elcoor = nodes[els[1, -4:], 1:3]
    kloc, _ = uel.elast_quad4(elcoor, params)

    sensi_number = (np.dot(UC[els[:,-4:]].reshape(nx*ny,8),kloc) * UC[els[:,-4:]].reshape(nx*ny,8) ).sum(1)

    return sensi_number

def sensitivity_nodes(nodes, adj_nodes, centers, sensi_els):
    """
    Calculate the sensitivity of each node.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    adj_nodes : ndarray
        Adjacency matrix of nodes
    centers : ndarray
        Array with center of elements
    sensi_els : ndarra
        Sensitivity of each element without filter
        
    Returns
    -------
    sensi_nodes : ndarray
        Sensitivity of each nodes
    """
    sensi_nodes = []
    for n in nodes:
        connected_els = adj_nodes[int(n[0])]
        if connected_els.shape[0] > 1:
            delta = centers[connected_els] - n[1:3]
            r_ij = np.linalg.norm(delta, axis=1) # We can remove this line and just use a constant because the distance is always the same
            w_i = 1/(connected_els.shape[0] - 1) * (1 - r_ij/r_ij.sum())
            sensi = (w_i * sensi_els[connected_els]).sum(axis=0)
        else:
            sensi = sensi_els[connected_els[0]]
        sensi_nodes.append(sensi)
    sensi_nodes = np.array(sensi_nodes)

    return sensi_nodes

def sensitivity_filter(nodes, centers, sensi_nodes, r_min):
    """
    Performe the sensitivity filter.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    sensi_nodes : ndarray
        Array with nodal sensitivity
    centers : ndarray
        Array with center of elements
    r_min : ndarra
        Minimum distance 
        
    Returns
    -------
    sensi_els : ndarray
        Sensitivity of each element with filter
    """
    sensi_els = []
    for i, c in enumerate(centers):
        delta = nodes[:,1:3]-c
        r_ij = np.linalg.norm(delta, axis=1)
        omega_i = (r_ij < r_min)
        w = 1/(omega_i.sum() - 1) * (1 - r_ij[omega_i]/r_ij[omega_i].sum())
        sensi_els.append((w*sensi_nodes[omega_i]).sum()/w.sum())
        
    sensi_els = np.array(sensi_els)
    sensi_els = sensi_els/sensi_els.max()

    return sensi_els