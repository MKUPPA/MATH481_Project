"""
# Author: Mridula Kuppa

# Description: Main file to solve heat equation on unit sphere
with IC1: Soccer-ball function

"""

import sys
import numpy as np
import utils as ul
import plot_utils as pu

# %% Creat X,Y grid
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
X,Y,Z = ul.chart2sph(X, Y, hemi='top')

# %% Get initial solution on sphere
sol_lst = ul.get_init_sol(X,Y,Z)
init_sol = sol_lst[0]

#------------------------------------------------------------------------------
# %% Plots

# # 2D grid with ghost cells
# savename = 'plots/2Dgrid.png'
# pu.plot_grid(X, Y, ghost_cells, savename=None, showplot=True)

# Analytical sol on 2D grid
# tval = ['0p00', '0p25', '0p50', '0p75', '1p00']
# for i, sol in enumerate(sol_lst):
    
#     savename = f'plots/ana{tval[i]}.png'
#     if i < len(sol_lst)-1:
#         pu.plot_sol(X, Y, sol, cbar=True, savename=savename, showplot=True)
#     else:
#         pu.plot_sol(X, Y, sol, cbar=True, savename=savename, showplot=True)
        
# Analytical_sol on sphere

# Nc = 101
# Xc, Yc, dxc, dyc = ul.get_cart_grid(Nc)

# Xu,Yu,Zu = ul.chart2sph(Xc, Yc, hemi='top')
# Xl,Yl,Zl = ul.chart2sph(Xc, Yc, hemi='bottom')

# Xtot = np.concatenate((Xu, Xl))
# Ytot = np.concatenate((Yu, Yl))
# Ztot = np.concatenate((Zu, Zl))

# sol_up = ul.get_init_sol(Xu,Yu,Zu)
# sol_low = ul.get_init_sol(Xl,Yl,Zl)

# init_up = sol_up[0]
# init_low = sol_low[0]

# init_tot = np.concatenate((init_up, init_low))

# pu.plot_sol_sph(Xtot, Ytot, Ztot, init_tot, cbar = True, savename='plots/IC_analytical/sphere_IC1.png', showplot=True)

#-------------------------------------------------------------------------------------------

# %% Extrapolate IC to ghost cells
ghost_sol = ul.extrapolate_ghost(init_sol, dx, dy, int_dict, ghost_dict)

# %% Global IC
sol = np.concatenate((init_sol, ghost_sol))

# %% Time integration

dt = 0.002
alpha = 1/42
nt = 500

# print(alpha*dt/(dx**2))

for i in range(nt):
    
    print(f'Time step {i}')
    
    if i == 0:
        sol_n = sol
        sol_int = init_sol
        
    # Compute finite differences
    dfdx, dfdy, d2fdx2, d2fdy2, d2fdxdy = ul.get_fd(X, Y, global_dict, sol_n, dx, dy)
    
    # Inverse metric components
    g11, g12, g21, g22, dg11_dx1, dg22_dx2, dg12_dx1, dg21_dx2, det_g = ul.inv_metric(X, Y)
    
    # LB operator at interior points
    L = ((dfdx*dg11_dx1 + np.sqrt(det_g)*g11*d2fdx2) + (-dfdy*dg12_dx1 + np.sqrt(det_g)*g12*d2fdxdy) + \
        (-dfdx*dg21_dx2 + np.sqrt(det_g)*g21*d2fdxdy) + (dfdy*dg22_dx2 + np.sqrt(det_g)*g22*d2fdy2))/np.sqrt(det_g)
    
    # Compute solution at next time step
    sol_np1 = (alpha*dt*L) + sol_int
    
    # Extrapolate to ghost cells
    ghost_sol = ul.extrapolate_ghost(sol_np1, dx, dy, int_dict, ghost_dict)
    
    # Concatenate ghost cells and interior; update solution
    sol_n = np.concatenate((sol_np1, ghost_sol))
    sol_int = sol_np1
    
    # plot

    if i == 124:
        np.savetxt('ic1_sol0p25.txt', sol_np1)
        savename = 'plots/num0p25.png'
        pu.plot_sol(X, Y, sol_np1, cbar=False, savename=savename, showplot=True)
        
    if i == 249:
        np.savetxt('ic1_sol0p50.txt', sol_np1)
        savename = 'plots/num0p50.png'
        pu.plot_sol(X, Y, sol_np1, cbar=False, savename=savename, showplot=False)
        
    if i == 374:
        np.savetxt('ic1_sol0p75.txt', sol_np1)
        savename = 'plots/num0p75.png'
        pu.plot_sol(X, Y, sol_np1, cbar=False, savename=savename, showplot=False)
        
    if i == 499:
        np.savetxt('ic1_sol1p00.txt', sol_np1)
        savename = 'plots/num1p00.png'
        pu.plot_sol(X, Y, sol_np1, cbar=False, savename=savename, showplot=False)

