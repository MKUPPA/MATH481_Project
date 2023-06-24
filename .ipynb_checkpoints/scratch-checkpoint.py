import numpy as np
import matplotlib.pyplot as plt

x = [1,1,0,-1,-1,-1,0,1]
y = [0,1,1,1,0,-1,-1,-1]
d = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]

for i in range(len(x)):
    
    a = x[i]
    b = y[i]
    
    if a == 0 and b>0:
        theta = np.pi/2
        
    elif a == 0 and b<0:
        theta = 3*np.pi/2
        
    elif a<0 and b>=0:
        theta = np.arctan(b/a) + np.pi
        
    elif a<0 and b<0:
        theta = np.arctan(b/a) + np.pi
    
    elif a>0 and b<0:
        theta = np.arctan(b/a) + 2*np.pi

    else:
        theta = np.arctan(b/a) 

    print(f'x = {a}, y = {b}, computed = {theta}, desired = {d[i]}')
    

# dict1 = {}

# dict1[0] = [1,2]

# vals = dict1.values()
# print(vals)

# ln = [1,2]

# if ln in vals:
#     print('true')

# dict1[1] = [2,5]

# vals = dict1.values()
# print(vals)

# ln = [2,3]

# if ln in vals:
#     print('true')
    
# else:
#     dict1[2] = ln
    
# vals = np.array(list(dict1.values()))

# print(vals[:,0])

# print(len(dict1))

# # plt.figure()

# # for value in vals:

# #     plt.scatter(value[0], value[1])

# # plt.show()


# new_dict = {}

# for i, val in enumerate(vals):
    
#     new_dict[i] = val
    
# print(new_dict)



# """

        
#         # 1st point, not on circle    
#         elif i==0:
            
#             dfdx[i] = (sol_n[i+1] - sol_n[i])/(dx)
            
            
#             # need left ghost cells value
            
#             # point on circle with same y-coord
#             x_circle = np.sqrt(1-y**2)
#             y_circle = y
#             sol_circle = point_init_sol(x_circle, y_circle)
            
#             dx1 = x - x_circle
#             dx2 = dx - dx1
            
#             left_neib_sol = sol_circle - (sol_n[i] - sol_circle)*(dx2/dx1) 
            
#             d2fdx2[i] = (sol_n[i+1] - 2*sol_n[i] + left_neib_sol)/(dx**2)

            
#         # last point, not on circle    
#         elif i==len(X):
#             dfdx[i] = (sol_n[i] - sol_n[i-1])/(dx)

#         # point has right neighbor
#         elif Y[i-1] != Y[i] and Y[i+1] == Y[i]:
#             #print(i, 'point has right neighbor')
#             dfdx[i] = (sol_n[i+1] - sol_n[i])/(dx)
            
            
#             # need left ghost cells value
            
#             # point on circle with same y-coord
#             x_circle = np.sqrt(1-y**2)
#             y_circle = y
#             sol_circle = point_init_sol(x_circle, y_circle)
            
#             dx1 = x - x_circle
#             dx2 = dx - dx1
            
#             left_neib_sol = sol_circle - (sol_n[i] - sol_circle)*(dx2/dx1) 
            
#             d2fdx2[i] = (sol_n[i+1] - 2*sol_n[i] + left_neib_sol)/(dx**2)
            

#         # point has left neighbor
#         elif Y[i+1] != Y[i] and Y[i-1] == Y[i]:
#             #print(i, 'point has left neighbor')
#             dfdx[i] = (sol_n[i] - sol_n[i-1])/(dx)

#         else:
#             dfdx[i] = (sol_n[i+1] - sol_n[i-1])/(2*dx)
#             d2fdx2[i] = (sol_n[i+1] - 2*sol_n[i] + sol_n[i-1])/(dx**2)

            
    
#     #---------------------------------------------------------------
#     """Computing dfdy"""

            
#     for i,(x,y) in enumerate(zip(Y, X)):
    
#         """
#         i=0 : Either on circle (if N is odd) or will have a top neighbor (N is even)
#         i=end: Either on circle (if N is odd) or will have a bottom neighbor
        
#         """
    
#         # Y_circle now gives the x-coordinate

#         point_ind = get_keys_from_value(point_dict, [x, y])[0]

#         if x**2 + y**2 == 1:
#             #print(i, 'point on circle')
#             dfdy[point_ind] = 0
#             continue
            
#         # 1st point, not on circle    
#         elif i==0:
            
#             # get index of top neighbor
#             ind_top = get_keys_from_value(point_dict, [Y[i], X[i+1]])[0]
            
#             dfdy[point_ind] = (sol_n[ind_top] - sol_n[i])/(dy)
            
#         # last point, not on circle    
#         elif i==len(X):
            
#             # get index of bot neighbor
#             ind_bot = get_keys_from_value(point_dict, [Y[i], X[i-1]])[0]
            
#             dfdy[point_ind] = (sol_n[i] - sol_n[ind_bot])/(dy)

#         # point has top neighbor    
#         elif Y[i-1] != Y[i] or Y[i+1] == Y[i]:
#             #print(i, 'point has top neighbor')
            
#             # get index of top neighbor
#             ind_top = get_keys_from_value(point_dict, [Y[i], X[i+1]])[0]
            
#             dfdy[point_ind] = (sol_n[ind_top] - sol_n[i])/(dy)

#         # point has bottom neighbor
#         elif Y[i+1] != Y[i] or Y[i-1] == Y[i]:
#             #print(i, 'point has bot neighbor')
            
#             # get index of bot neighbor
#             ind_bot = get_keys_from_value(point_dict, [Y[i], X[i-1]])[0]
            
#             dfdy[point_ind] = (sol_n[i] - sol_n[ind_bot])/(dy)

#         else:
            
#             # get indices of neighbors
#             ind_top = get_keys_from_value(point_dict, [Y[i], X[i+1]])[0]
#             ind_bot = get_keys_from_value(point_dict, [Y[i], X[i-1]])[0]

#             dfdy[point_ind] = (sol_n[ind_top] - sol_n[ind_bot])/(2*dy)
            
            
#     #---------------------------------------------------------------------
    


# """