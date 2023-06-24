# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:09:12 2023

@author: kuppa
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
import utils as ul


"""
phi - colatitude
theta - longitude
scipy.special.sph_harm(upper, lower, long, lat)

"""

#==========================================

# phi = np.linspace(0, np.pi, 100)
# theta = np.linspace(0, 2*np.pi, 100) 

# phi, theta = np.meshgrid(phi, theta)

# # The Cartesian coordinates of the unit sphere
# x = np.sin(phi) * np.cos(theta)
# y = np.sin(phi) * np.sin(theta)
# z = np.cos(phi)

#=========================================

# %% Creat X,Y grid
N = 101
X, Y, dx, dy = ul.get_cart_grid(N)
X,Y,Z = ul.chart2sph(X, Y, hemi='top')
theta, phi = ul.cart2sph(X, Y, Z)
x = X
y = Y
z = Z
    

# Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
t1 = (sph_harm(0, 6, theta, phi)).real
# fmax, fmin = t1.max(), t1.min()
# t1 = (t1 - fmin)/(fmax - fmin)

t2 = (sph_harm(5, 6, theta, phi)).real 
print(np.shape(t2))

# fmax, fmin = t2.max(), t2.min()
# t2 = (t2 - fmin)/(fmax - fmin)

fcolors = np.sqrt(14/11)*t2 + t1

alpha = 1/42
t1 = 0.25
t2 = 1.00
sol_t1 = np.exp(-42*alpha*t1)*fcolors
sol_t2 = np.exp(-42*alpha*t2)*fcolors


fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(1,3,1,projection='3d')
ax2 = fig.add_subplot(1,3,2,projection='3d')
ax3 = fig.add_subplot(1,3,3,projection='3d')

sc1 = ax1.scatter(x, y, z, s=1, c=fcolors, cmap=cm.jet, vmin=-1, vmax=1.5)
sc2 = ax2.scatter(x, y, z, s=1, c=sol_t1, cmap=cm.jet, vmin=-1, vmax=1.5)
sc3 = ax3.scatter(x, y, z, s=1, c=sol_t2, cmap=cm.jet, vmin=-1, vmax=1.5)
plt.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(sc1, cax=cbar_ax)
plt.show()

plt.figure(figsize=(5,4))
sc = plt.scatter(x, y, s=1, c = fcolors, cmap=cm.jet, vmin=-1, vmax=1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()