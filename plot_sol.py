"""
code to plot saved solutions

"""
import numpy as np
import utils as ul
import plot_utils as pu

# %% IC1
N = 101
X, Y, dx, dy = ul.get_cart_grid(N)

files = ['sol0p25.txt', 'sol0p50.txt', 'sol0p75.txt', 'sol1p00.txt']
names = ['num0p25.png', 'num0p50.png', 'num0p75.png', 'num1p00.png']

for i, file in enumerate(files):

    sol_file = 'Solutions/IC1/' + file
    u = np.loadtxt(sol_file)

    savename = 'plots/' + names[i]
    pu.plot_sol(X, Y, u, cbar=True, savename=savename, showplot=True)

    
