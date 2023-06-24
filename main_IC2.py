"""
# Author: Mridula Kuppa

# Description: Main file to solve heat equation on unit sphere
with IC2: Hot upper hemisphere (u=1), cold lower hemisphere (u=0)

"""

import sys
import numpy as np
import utils as ul
import plot_utils as pu
import scipy

# %% Create X,Y grid
N = 51
X, Y, dx, dy = ul.get_cart_grid(N)

# %% Interior point dictionary
int_dict = ul.get_interior_dict(X, Y)
vals = int_dict.values()

# %% Initialize ghost cells
ghost_dict = ul.get_ghost(X, Y, dx, dy)
ghost_cells = np.array(list(ghost_dict.values()))    

# %% Global dictionary
global_dict = ul.get_global_dict(int_dict, ghost_dict)
global_points = np.array(list(global_dict.values()))

# %% Project back to sphere
Xt,Yt,Zt = ul.chart2sph(X, Y, hemi='top')
Xb,Yb,Zb = ul.chart2sph(X, Y, hemi='bottom')

# %% Get initial solution on sphere
sol_lst = ul.get_init_sol(Xt,Yt,Zt, case='hot_cold')
init_sol_t = sol_lst[0]

sol_lst = ul.get_init_sol(Xb,Yb,Zb, case='hot_cold')
init_sol_b = sol_lst[0]

#-------------------------------------------------------------------------------------------------------

# %% Plot Initial solution on 2D grid
#pu.plot_sol(X, Y, init_sol_t, cbar=True, savename='init_top.png', showplot=True)
#pu.plot_sol(X, Y, init_sol_b, cbar=True, savename='init_bot.png', showplot=True)

# %% Plot Initial sol on sphere
# Nc = 101
# Xc, Yc, dxc, dyc = ul.get_cart_grid(Nc)

# Xu,Yu,Zu = ul.chart2sph(Xc, Yc, hemi='top')
# Xl,Yl,Zl = ul.chart2sph(Xc, Yc, hemi='bottom')

# Xtot = np.concatenate((Xu, Xl))
# Ytot = np.concatenate((Yu, Yl))
# Ztot = np.concatenate((Zu, Zl))

# sol_up = ul.get_init_sol(Xu,Yu,Zu,case='hot_cold')
# sol_low = ul.get_init_sol(Xl,Yl,Zl,case='hot_cold')

# init_up = sol_up[0]
# init_low = sol_low[0]

# init_tot = np.concatenate((init_up, init_low))

# pu.plot_sol_sph(Xtot, Ytot, Ztot, init_tot, cbar = True, savename='plots/IC_HotCold/sphere_IC2.png', showplot=True)

#----------------------------------------------------------------------------------------------------------------------

# %% Extrapolate IC to ghost cells
ghost_sol = ul.extrapolate_ghost(init_sol_t, dx, dy, int_dict, ghost_dict)

# %% Global IC
sol_t = np.concatenate((init_sol_t, ghost_sol))
sol_b = np.concatenate((init_sol_b, ghost_sol))

# %% Time integration

dt = 0.01
alpha = 1/42
nt = 2000

print(alpha*dt/(dx**2))

# Some necessary Matrices
I = np.eye(len(int_dict), dtype=int)
O = np.zeros((len(int_dict), len(int_dict)))
rhs_ghost = np.zeros((len(ghost_dict), ))

 # System of equations
A1 = np.hstack((I, O))
A2 = np.hstack((O, I))
contr_mat = ul.ghost_constraint(dx, dy, int_dict, ghost_dict)
A = np.vstack((A1, A2, contr_mat))

for i in range(nt):
    
    print(f'Time step {i}')
    
    if i == 0:
        
        # sol in top half
        sol_t_n = sol_t
        sol_t_int = init_sol_t
        
        # sol in bottom half
        sol_b_n = sol_b
        sol_b_int = init_sol_b
    
    # Inverse metric components
    g11, g12, g21, g22, dg11_dx1, dg22_dx2, dg12_dx1, dg21_dx2, det_g = ul.inv_metric(X, Y)
    
    # Computations in top half
    dfdx, dfdy, d2fdx2, d2fdy2, d2fdxdy = ul.get_fd(Xt, Yt, global_dict, sol_t_n, dx, dy)
    
    L = ((dfdx*dg11_dx1 + np.sqrt(det_g)*g11*d2fdx2) + (-dfdy*dg12_dx1 + np.sqrt(det_g)*g12*d2fdxdy) + \
        (-dfdx*dg21_dx2 + np.sqrt(det_g)*g21*d2fdxdy) + (dfdy*dg22_dx2 + np.sqrt(det_g)*g22*d2fdy2))/np.sqrt(det_g)
    
    rhs_t = (alpha*dt*L) + sol_t_int
    rhs_t = np.reshape(rhs_t, (len(rhs_t), 1))
    
    # Computations in bottom half
    dfdx, dfdy, d2fdx2, d2fdy2, d2fdxdy = ul.get_fd(Xb, Yb, global_dict, sol_b_n, dx, dy)
    
    L = ((dfdx*dg11_dx1 + np.sqrt(det_g)*g11*d2fdx2) + (-dfdy*dg12_dx1 + np.sqrt(det_g)*g12*d2fdxdy) + \
        (-dfdx*dg21_dx2 + np.sqrt(det_g)*g21*d2fdxdy) + (dfdy*dg22_dx2 + np.sqrt(det_g)*g22*d2fdy2))/np.sqrt(det_g)
    
    rhs_b = (alpha*dt*L) + sol_b_int
    rhs_b = np.reshape(rhs_b, (len(rhs_b), 1))

    rhs_ghost = np.reshape(rhs_ghost, (len(rhs_ghost), 1))
    b = np.vstack((rhs_t, rhs_b, rhs_ghost))
    
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    
    sol_t_np1 = x[0:len(int_dict)]
    sol_b_np1 = x[len(int_dict):]
    

    sol_t_np1 = np.reshape(sol_t_np1, (-1, ))
    sol_b_np1 = np.reshape(sol_b_np1, (-1, ))
    
    
    # Extrapolate to ghost cells
    ghost_sol = ul.extrapolate_ghost(sol_t_np1, dx, dy, int_dict, ghost_dict)
    
    # Concatenate ghost cells and interior; update solution
    sol_t_n = np.concatenate((sol_t_np1, ghost_sol))
    sol_b_n = np.concatenate((sol_b_np1, ghost_sol))
    
    sol_t_int = sol_t_np1
    sol_b_int = sol_b_np1
    
    # plot
        
    if i == 500: 
        np.savetxt('ic2_sol_t_5s.txt', sol_t_np1)
        savename = 'plots/ic2_num_t_5s.png'
        pu.plot_sol(X, Y, sol_t_int, cbar=True, savename=savename, showplot=False)
        
        np.savetxt('ic2_sol_b_5s.txt', sol_b_np1)
        savename = 'plots/ic2_num_b_5s.png'
        pu.plot_sol(X, Y, sol_b_int, cbar=True, savename=savename, showplot=False)
        
        
    if i == 1000: 
        np.savetxt('ic2_sol_t_10s.txt', sol_t_np1)
        savename = 'plots/ic2_num_t_10s.png'
        pu.plot_sol(X, Y, sol_t_np1, cbar=True, savename=savename, showplot=False)

        np.savetxt('ic2_sol_b_10s.txt', sol_b_np1)
        savename = 'plots/ic2_num_b_10s.png'
        pu.plot_sol(X, Y, sol_b_np1, cbar=True, savename=savename, showplot=False)
        
    if i == 1500: 
        np.savetxt('ic2_sol_t_15s.txt', sol_t_np1)
        savename = 'plots/ic2_num_t_15s.png'
        pu.plot_sol(X, Y, sol_t_np1, cbar=True, savename=savename, showplot=False)

        np.savetxt('ic2_sol_b_15s.txt', sol_b_np1)
        savename = 'plots/ic2_num_b_15s.png'
        pu.plot_sol(X, Y, sol_b_np1, cbar=True, savename=savename, showplot=False)
        
    if i == 1999: 
        np.savetxt('ic2_sol_t_20s.txt', sol_t_np1)
        savename = 'plots/ic2_num_t_20s.png'
        pu.plot_sol(X, Y, sol_t_np1, cbar=True, savename=savename, showplot=False)

        np.savetxt('ic2_sol_b_20s.txt', sol_b_np1)
        savename = 'plots/ic2_sol_b_20s.png'
        pu.plot_sol(X, Y, sol_b_np1, cbar=True, savename=savename, showplot=False)
        

