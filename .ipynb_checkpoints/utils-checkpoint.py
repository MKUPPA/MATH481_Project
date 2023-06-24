import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.special import sph_harm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection as p3dc
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

"""
Notation
--------------

theta - colatitude, [0,pi]
phi - azimuthal angle, [0,2pi]

"""

num_deci = 6

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


def get_interior_dict(X, Y):
    
    """
    Returns the interior point dictionary
    """
    
    # dictionary with point index as key and (x,y) as value
    point_ind = np.arange(0,len(X),1)
    point_dict = dict.fromkeys(point_ind)

    for i in range(len(X)):
    
        point_dict[i] = [X[i], Y[i]]
             
    return point_dict

# Extract ghost cells

def get_ghost(X, Y, dx, dy):
    
    
    int_dict = get_interior_dict(X, Y)
    int_values = int_dict.values()
    
    ghost_dict = {}
    ghost_values = ghost_dict.values()

    count = 0
    
    # when computing dfdx, d2fdx2
    for i,(x,y) in enumerate(zip(X, Y)):

        # left neigh
        ln = [np.round(x-dx, num_deci), y]

        if ln not in int_values and ln not in ghost_values:

            ghost_dict[count] = ln
            count = count+1
            ghost_values = ghost_dict.values()

        # right neigh
        rn = [np.round(x+dx, num_deci), y]

        if rn not in int_values and rn not in ghost_values:

            ghost_dict[count] = rn
            count = count+1
            ghost_values = ghost_dict.values()

        # top_neigh
        tn = [x, np.round(y+dy, num_deci)]

        if tn not in int_values and tn not in ghost_values:

            ghost_dict[count] = tn
            count = count+1
            ghost_values = ghost_dict.values()

        # bot neigh
        bn = [x, np.round(y-dy, num_deci)]

        if bn not in int_values and bn not in ghost_values:

            ghost_dict[count] = bn
            count = count+1
            ghost_values = ghost_dict.values()

        # upper diag neigh
        du = [np.round(x+dx, num_deci), np.round(y+dy, num_deci)]

        if du not in int_values and du not in ghost_values:

            ghost_dict[count] = du
            count = count+1
            ghost_values = ghost_dict.values()

        # lower diag neigh
        db = [np.round(x-dx, num_deci), np.round(y-dy, num_deci)]

        if db not in int_values and db not in ghost_values:

            ghost_dict[count] = db
            count = count+1
            ghost_values = ghost_dict.values()
    
    return ghost_dict

# Global dictionary

def get_global_dict(int_dict, ghost_dict):
    
    int_points = int_dict.values()
    ghost_points = ghost_dict.values()
    
    int_tot = int(len(int_points))
    
    global_dict = {}
    
    # add interior points
    for i, val in enumerate(int_points):
        
        global_dict[i] = val
        
    # add ghost points
    for i, val in enumerate(ghost_points):
        
        global_dict[int_tot + i] = val
    
    return global_dict

def extrapolate_ghost(sol_n, dx, dy, int_dict, ghost_dict):
    
    """
    Extrapolate solution to ghost cells
    
    Inputs
    ---------------
    
    sol_n: solution at interior points at nth time step
    ghost_dict
    
    """

    int_values = int_dict.values()
    ghost_sol = np.zeros((len(ghost_dict),))
    
    # list of points with no interior neighbors
    no_neib = []
    
    
    for i in ghost_dict:
        
        x,y = ghost_dict[i]
        
        # neighbors
        ln = [np.round(x-dx, num_deci), y]
        rn = [np.round(x+dx, num_deci), y]
        tn = [x, np.round(y+dy, num_deci)]
        bn = [x, np.round(y-dy, num_deci)]
        du = [np.round(x+dx, num_deci), np.round(y+dy, num_deci)]
        db = [np.round(x-dx, num_deci), np.round(y-dy, num_deci)]
        
        if ln in int_values:
            
            #print(i, '-> ln')
            # get index of left neigh and left-left neigh
            ind1 = get_keys_from_value(int_dict, ln)[0]
            ind2 = get_keys_from_value(int_dict, [np.round(x-2*dx, num_deci),y])[0]
            ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
            
        elif rn in int_values:
            
            
            #print(i, '-> rn')
            ind1 = get_keys_from_value(int_dict, rn)[0]
            ind2 = get_keys_from_value(int_dict, [np.round(x+2*dx, num_deci),y])[0]
            ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
            
        elif tn in int_values:
            
            #print(i, '-> tn')
            ind1 = get_keys_from_value(int_dict, tn)[0]
            ind2 = get_keys_from_value(int_dict, [x,np.round(y+2*dy, num_deci)])[0]
            ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
            
        elif bn in int_values:
            
            #print(i, '-> bn')
            ind1 = get_keys_from_value(int_dict, bn)[0]
            ind2 = get_keys_from_value(int_dict, [x,np.round(y-2*dy, num_deci)])[0]
            ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
            
        
        elif du in int_values:
            
            #print(i, '-> du')
            ind1 = get_keys_from_value(int_dict, du)[0]
            ind2 = get_keys_from_value(int_dict, [np.round(x+2*dx, num_deci),np.round(y+2*dy, num_deci)])[0]
            ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
        
        elif db in int_values:
            
            #print(i, '-> db')
            ind1 = get_keys_from_value(int_dict, db)[0]
            ind2 = get_keys_from_value(int_dict, [np.round(x-2*dx, num_deci),np.round(y-2*dy, num_deci)])[0]
            ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
            
        else:
            
            no_neib.append(i)
            
    #print('No interior neighbor points = ')
    #print(no_neib)
            
    return ghost_sol

def ghost_constraint(dx, dy, int_dict, ghost_dict):
    
    """
    Extrapolate solution to ghost cells
    
    Inputs
    ---------------
    
    sol_n: solution at interior points at nth time step
    ghost_dict
    
    """

    int_values = int_dict.values()
    
    N1 = len(ghost_dict)
    N2 = len(int_dict)
    
    contr_mat = np.zeros((N1, 2*N2))  
    
    count = 0
        
    for i in ghost_dict:
        
        x,y = ghost_dict[i]
        
        # neighbors
        ln = [np.round(x-dx, num_deci), y]
        rn = [np.round(x+dx, num_deci), y]
        tn = [x, np.round(y+dy, num_deci)]
        bn = [x, np.round(y-dy, num_deci)]
        du = [np.round(x+dx, num_deci), np.round(y+dy, num_deci)]
        db = [np.round(x-dx, num_deci), np.round(y-dy, num_deci)]
        
        if ln in int_values:
            
            ind1 = get_keys_from_value(int_dict, ln)[0]
            ind2 = get_keys_from_value(int_dict, [np.round(x-2*dx, num_deci),y])[0]
            
            #ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
            
            contr_mat[count, ind1] = 2
            contr_mat[count, ind2] = -1
            contr_mat[count, ind1 + N2] = -2
            contr_mat[count, ind2 + N2] = 1 
            
            count = count + 1
            
        elif rn in int_values:
            
            ind1 = get_keys_from_value(int_dict, rn)[0]
            ind2 = get_keys_from_value(int_dict, [np.round(x+2*dx, num_deci),y])[0]
            
            #ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
            
            contr_mat[count, ind1] = 2
            contr_mat[count, ind2] = -1
            contr_mat[count, ind1 + N2] = -2
            contr_mat[count, ind2 + N2] = 1
            count = count + 1
            
        elif tn in int_values:
            
            ind1 = get_keys_from_value(int_dict, tn)[0]
            ind2 = get_keys_from_value(int_dict, [x,np.round(y+2*dy, num_deci)])[0]

            #ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
            
            contr_mat[count, ind1] = 2
            contr_mat[count, ind2] = -1
            contr_mat[count, ind1 + N2] = -2
            contr_mat[count, ind2 + N2] = 1
            
            count = count + 1
            
        elif bn in int_values:
            
            ind1 = get_keys_from_value(int_dict, bn)[0]
            ind2 = get_keys_from_value(int_dict, [x,np.round(y-2*dy, num_deci)])[0]

            #ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
            
            contr_mat[count, ind1] = 2
            contr_mat[count, ind2] = -1
            contr_mat[count, ind1 + N2] = -2
            contr_mat[count, ind2 + N2] = 1
            
            count = count + 1
            
        
        elif du in int_values:
            
            ind1 = get_keys_from_value(int_dict, du)[0]
            ind2 = get_keys_from_value(int_dict, [np.round(x+2*dx, num_deci),np.round(y+2*dy, num_deci)])[0]
            
            #ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
        
            contr_mat[count, ind1] = 2
            contr_mat[count, ind2] = -1
            contr_mat[count, ind1 + N2] = -2
            contr_mat[count, ind2 + N2] = 1
            
            count = count + 1
            
        
        elif db in int_values:
            
            ind1 = get_keys_from_value(int_dict, db)[0]
            ind2 = get_keys_from_value(int_dict, [np.round(x-2*dx, num_deci),np.round(y-2*dy, num_deci)])[0]
            
            #ghost_sol[i] = 2*sol_n[ind1] - sol_n[ind2]
            
            contr_mat[count, ind1] = 2
            contr_mat[count, ind2] = -1
            contr_mat[count, ind1 + N2] = -2
            contr_mat[count, ind2 + N2] = 1
            
            count = count + 1
            
            
        else:
            
            no_neib.append(i)
            
    return contr_mat

def cart2sph(X,Y,Z):
    
    """
    Returns spherical coordinates given cartesian
    in radians
    
    theta - colatitude
    phi - azimuthal
    
    """
    
    theta = np.zeros_like(X)
    phi = np.zeros_like(X)

    
    for i,(x,y,z) in enumerate(zip(X, Y, Z)):
    
        # Unit sphere
        r = np.sqrt(x**2 + y**2 + z**2)
    

        if x == 0 and y>=0:
            phi[i] = np.pi/2
        
        elif x == 0 and y<0:
            phi[i] = 3*np.pi/2
        
        elif x<0 and y>=0:
            phi[i] = np.arctan(y/x) + np.pi
        
        elif x<0 and y<0:
            phi[i] = np.arctan(y/x) + np.pi
    
        elif x>0 and y<0:
            phi[i] = np.arctan(y/x) + 2*np.pi

        else:
            phi[i] = np.arctan(y/x) 

        
        theta[i] = np.arctan(np.sqrt(x**2 + y**2)/z)
        
    
    return theta, phi

def sph2cart(theta, phi):
    
    """
    Returns cartesian coordinates given colatitude(theta) and azimuth(phi)
    input in radians
    """
    
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    
    return x,y,z

def sph2chart(x,y,z):
    
    """
    Coordinate chart projection
    """
    x1 = x
    x2 = y
    
    return x1,x2


def chart2sph(x1, x2, hemi):
    
    if hemi == 'top':
    
        z = np.sqrt(1 - x1**2 - x2**2)
        
    else:
        
        z = -np.sqrt(1 - x1**2 - x2**2)
    
    return x1,x2,z

def metric(x1,x2):
    
    """
    Returns components and determinant of metric tensor
    """
    
    g11 = 1 + (x1**2/(x1**2 + x2**2))
    g12 = x1*x2/(x1**2 + x2**2)
    g21 = x1*x2/(x1**2 + x2**2)
    g22 = 1 + (x2**2/(x1**2 + x2**2))
    
    det = g11*g22 - g12*g21
    
    return g11, g12, g21, g22, det

    
def inv_metric(x1, x2):
    
    epsilon = 1e-8
    
    # determinant of metric tensor
    
    det_g = 1/(1 - x1**2 - x2**2)
    
    g11 = (1 + ((x2**2)/(1 - x1**2 - x2**2)))/det_g
    g12 = -(x1*x2/(1 - x1**2 - x2**2))/det_g
    g21 = -(x1*x2/(1 - x1**2 - x2**2))/det_g
    g22 = (1 + ((x1**2)/(1 - x1**2 - x2**2)))/det_g
    
    dg11_dx1 = (-2*x1*(1 - x1**2 - x2**2) + x1*(1 - x1**2))/((1 - x1**2 - x2**2)**(3/2))
    dg22_dx2 = (-2*x2*(1 - x2**2 - x1**2) + x2*(1 - x2**2))/((1 - x2**2 - x1**2)**(3/2))
    dg12_dx1 = x2*(1 - x2**2)/((1 - x2**2 - x1**2)**(3/2))
    dg21_dx2 = x1*(1 - x1**2)/((1 - x2**2 - x1**2)**(3/2))

    
    return g11, g12, g21, g22, dg11_dx1, dg22_dx2, dg12_dx1, dg21_dx2, det_g


# Define a function to compute IC

def get_init_sol(X,Y,Z,case='ic_ana'):
    
    theta_grid, phi_grid = cart2sph(X, Y, Z)
    

    x = X
    y = Y
    z = Z
    
    if case == 'ic_ana':
        
        # %% -------------IC with analytical solution ---------------------

        """scipy.special.sph_harm(order, deg, azimuth, colat, out=None)"""
    
        term1 = (sph_harm(5, 6, phi_grid, theta_grid)).real
        term2 = (sph_harm(0, 6, phi_grid, theta_grid)).real
        init_sol = np.sqrt(14/11)*term1 + term2
        alpha = 1/42
            
        t_lst = [0.25, 0.5, 0.75, 1.00]

        sol_lst = [init_sol]
        for t in t_lst:
    
            sol_t = np.exp(-42*alpha*t)*init_sol
            sol_lst.append(sol_t)
            
    elif case == 'hot_cold':
        
         
         # %% --------------- IC with top 1, bottom 0 ----------------
        init_sol = np.full_like(X, 0)
        
        #init_sol[:] = 1
        
        for i,val in enumerate(Z):
            if val >= 0:
                init_sol[i] = 1
            else:
                init_sol[i] = 0
        #n = X.shape[0]
        
        #init_sol[0:int(n/2)] = 1
        #init_sol[int(n/2):] = 0
        
        sol_lst = [init_sol]
    
    return sol_lst

# generate grid on XY plane

def get_cart_grid(N):
    
    """
    Returns cartesian grid inside a unit circle
    
    Inputs
    -------
    
    N - No.of grid points
    """

    # Create an array of x and y coordinates within a square bounding the unit circle
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    
    dx = np.round(x[1] - x[0], num_deci)
    dy = np.round(y[1] - y[0], num_deci)

    X, Y = np.meshgrid(x, y)
    
    # Filter the coordinates that fall within the unit circle
    mask = (X**2 + Y**2) < 1
    
    # one dimen ndarrays
    X_circle = np.round(X[mask], num_deci)
    Y_circle = np.round(Y[mask], num_deci)
    
    return X_circle, Y_circle, dx, dy
    


# Generate finite differences

def get_fd(X, Y, global_dict, global_sol, dx, dy):
    
    """
    Inputs 
    --------------
    
    X - 1d array of all x-coords
    Y - 1d array of all x-coords
    global_dict - contains int_points and ghost_points
    global_sol_n - includes solution at int_points and ghost_points
    dx
    dy
    
    Outputs
    ----------
    First order, 2nd order and mixed derivatives at interior points
    
    """
        
    # Initialize arrays to hold the values of the gradients
    dfdx = np.zeros_like(X)
    dfdy = np.zeros_like(Y)
    
    d2fdx2 = np.zeros_like(X)
    d2fdy2 = np.zeros_like(Y)
    
    d2fdxdy = np.zeros_like(X)
    
    for i,(x,y) in enumerate(zip(X, Y)):
            
        # get all neighbors
        ln = [np.round(x-dx, num_deci), y]
        rn = [np.round(x+dx, num_deci), y]
        tn = [x, np.round(y+dy, num_deci)]
        bn = [x, np.round(y-dy, num_deci)]
        du = [np.round(x+dx, num_deci), np.round(y+dy, num_deci)]
        db = [np.round(x-dx, num_deci), np.round(y-dy, num_deci)]
        
        # get indices of neighbors
        ind_ln = get_keys_from_value(global_dict, ln)[0]
        ind_rn = get_keys_from_value(global_dict, rn)[0]
        ind_tn = get_keys_from_value(global_dict, tn)[0]
        ind_bn = get_keys_from_value(global_dict, bn)[0]
        ind_du = get_keys_from_value(global_dict, du)[0]
        ind_db = get_keys_from_value(global_dict, db)[0]
        
        ind_p = get_keys_from_value(global_dict, [x,y])[0]

        # compute finite differences 
            
        dfdx[i] = (global_sol[ind_rn] - global_sol[ind_ln])/(2*dx)     
        dfdy[i] = (global_sol[ind_tn] - global_sol[ind_bn])/(2*dy)
        
        d2fdx2[i] = (global_sol[ind_rn] - 2*global_sol[ind_p] + global_sol[ind_ln])/(dx**2) 
        d2fdy2[i] = (global_sol[ind_tn] - 2*global_sol[ind_p] + global_sol[ind_bn])/(dy**2)
        
        d2fdxdy[i] = (global_sol[ind_du] + global_sol[ind_db] - \
                      global_sol[ind_rn] - global_sol[ind_ln] - global_sol[ind_tn] - global_sol[ind_bn] + \
                      2*global_sol[ind_p])/(2*dx*dy)    

    return dfdx, dfdy, d2fdx2, d2fdy2, d2fdxdy

