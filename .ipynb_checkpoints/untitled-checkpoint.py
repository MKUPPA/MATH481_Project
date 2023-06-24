"""
Solve heat equation on sphere
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

#--------------------------------------------------------------------
# x, y = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))

# print(np.shape(x))
# print(np.shape(y))

# plt.figure()
# plt.scatter(x, y)
# # plt.plot(radius*np.cos(theta), radius*np.sin(theta))
# plt.show()

# radius = 1
# theta = np.linspace(0, 2*np.pi, 100)
# mask = x**2 + y**2 <= radius

# x_new, y_new = x[mask], y[mask]


# plt.figure()
# plt.scatter(x_new, y_new)
# plt.plot(radius*np.cos(theta), radius*np.sin(theta))
# plt.show()

# plt.figure()
# for i in range(len(x_new)):
    
    
#     if i == 2:
        
#         print(x_new[i], y_new[i])
#         print(x_new[i+1], y_new[i+1])
#         print(x_new[i-1], y_new[i-1])
        
#         plt.scatter(x_new[i], y_new[i], label='xiyi')
#         plt.scatter(x_new[i+1], y_new[i+1], label='xi+1, yi+1')
#         plt.scatter(x_new[i-1], y_new[i-1], label='xi-1, yi-1')
        
# plt.legend()        
# plt.show()
            
#----------------------------------------------------------------------    
    
# Number of grid points in each direction
N = 101

# Create an array of x and y coordinates within a square bounding the unit circle
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Filter the coordinates that fall within the unit circle
mask = (X**2 + Y**2) <= 1

# one dimen ndarrays
X_circle = X[mask]
Y_circle = Y[mask]


# for i,(x,y) in enumerate(zip(Y_circle, X_circle)):
    
#     print(i,x,y)


# no.of cells with same y
y_unique = np.unique(Y_circle)
ny_unique = len(y_unique)
ny_dict = dict.fromkeys(y_unique)

for y in y_unique:
    
    cells = Y_circle == y
    ny_dict[y] = len(Y_circle[cells])
    
# dictionary with point index as key and (x,y) as value
point_ind = np.arange(0,len(X_circle),1)
point_dict = dict.fromkeys(point_ind)

for i in range(len(X_circle)):
    
    point_dict[i] = [X_circle[i], Y_circle[i]]

# function
def f(x,y):
    
    return 2*x


# Initialize arrays to hold the values of the gradients
dfdx = np.zeros_like(X_circle)
dfdy = np.zeros_like(Y_circle)
      
dx = 1
dy = 1
    
for i,(x,y) in enumerate(zip(X_circle, Y_circle)):
    
    
    """For computing dfdx"""
    
    if x**2 + y**2 == 1:
        #print(i, 'Point on circle')
        dfdx[i] = 0
        continue
    
    # point has right neighbor
    elif Y_circle[i-1] != Y_circle[i] and Y_circle[i+1] == Y_circle[i]:
        #print(i, 'point has right neighbor')
        dfdx[i] = (f(X_circle[i+1], Y_circle[i]) - f(X_circle[i], Y_circle[i]))/(dx)
     
    # point has left neighbor
    elif Y_circle[i+1] != Y_circle[i] and Y_circle[i-1] == Y_circle[i]:
        #print(i, 'point has left neighbor')
        dfdx[i] = (f(X_circle[i], Y_circle[i]) - f(X_circle[i-1], Y_circle[i]))/(dx)
        
    else:
        dfdx[i] = (f(X_circle[i+1], Y_circle[i]) - f(X_circle[i-1], Y_circle[i]))/(2*dx)
        
# print(dfdx)


for i,(x,y) in enumerate(zip(Y_circle, X_circle)):
    
    """For computing dfdy"""
    # Y_circle now gives the x-coordinate
    
    point_ind = get_keys_from_value(point_dict, [x, y])[0]
    
    if x**2 + y**2 == 1:
        #print(i, 'point on circle')
        dfdy[point_ind] = 1
        continue
        
    # point has top neighbor    
    elif Y_circle[i-1] != Y_circle[i] or Y_circle[i+1] == Y_circle[i]:
        #print(i, 'point has top neighbor')
        dfdy[point_ind] = (f(Y_circle[i], X_circle[i+1]) - f(Y_circle[i], X_circle[i]))/dy
        
    # point has bottom neighbor
    elif Y_circle[i+1] != Y_circle[i] or Y_circle[i-1] == Y_circle[i]:
        #print(i, 'point has bot neighbor')
        dfdy[point_ind] = (f(Y_circle[i], X_circle[i]) - f(Y_circle[i], X_circle[i-1]))/dy
        
    else:
       
        dfdy[point_ind] = (f(Y_circle[i], X_circle[i+1]) - f(Y_circle[i], X_circle[i-1]))/(2*dy)
        
# print(dfdy)    
        
plt.figure(figsize=(3,3))
plt.scatter(X_circle, Y_circle, s=1)
plt.show()      
        
sys.exit()
    

    
    

"""    
# rectangular grid

x = np.linspace(-1,1,11)
y = np.linspace(-1,1,11)
 
for i in range(len(x)):
    for j in range(len(y)):
        
        df_dy(i,j) = (f(x[i], y[j+1]) - f(x[i], y[j]))/dy
        df_dx(i,j) = (f(x[i+1], y[j]) - f(x[i], y[j]))/dx
        
"""



    