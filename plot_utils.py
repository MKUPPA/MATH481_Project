from matplotlib import cm
import matplotlib.pyplot as plt


def plot_grid(X, Y, ghost_cells, savename=None, showplot=None):
    
    # plot 2D grid with ghost cells
    plt.figure(figsize=(3,3))
    plt.scatter(X, Y, s=1)
    plt.scatter(ghost_cells[:,0], ghost_cells[:,1], c='r', s=1, marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if savename:
        plt.savefig(savename)
    if showplot:
        plt.show()

    return


def plot_sol(X, Y, sol, cbar=False, savename=None, showplot=None):
    
    # Plot sol on 2D grid
    if cbar:
        plt.figure(figsize=(5,5))
    else:
        plt.figure(figsize=(3,3))
        
    sc = plt.scatter(X, Y, s=1, c = sol, cmap=cm.jet, vmin=0, vmax=1)
    if cbar:
        plt.colorbar(sc,fraction=0.046, pad=0.04)
    #plt.xlabel('x')
    #plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    if savename:
        plt.savefig(savename)
    if showplot:
        plt.show()
        
    return

def plot_sol_sph(x,y,z,sol,cbar=True,savename=None,showplot=None):
    
    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_subplot(1,1,1,projection='3d')

    sc1 = ax1.scatter(x, y, z, s=1, c=sol, cmap=cm.jet, vmin=-1, vmax=1.5)
    ax1.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    if cbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(sc1, cax=cbar_ax)
    if savename:
        plt.savefig(savename)
    if showplot:
        plt.show()

    
    return

        

"""


    
    # Plot IC on sphere
    
    
    
    # If global
    
    # # Initial solution at interior points and ghost cells
# plt.figure(figsize=(4,4))
# sc = plt.scatter(global_points[:,0], global_points[:,1], s=1, c = sol, cmap=cm.jet)
# # plt.plot(np.cos(theta), np.sin(theta))
# plt.colorbar(sc)
# plt.show()
    
    return
    
    
def plot_num():
    
    
    # plot FD solution at required time steps
    
    if i == 124:
    
        plt.figure(figsize=(5,4))
        sc = plt.scatter(X, Y, s=1, c = sol_np1, cmap=cm.jet, vmin=-1, vmax=1.5)
        #plt.colorbar(sc)
        #plt.plot(np.cos(theta), np.sin(theta), 'k--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        #plt.savefig('plots/sol0p25_dt0p002.png')
        plt.show()
        
    if i == 249:
        
        plt.figure(figsize=(5,4))
        sc = plt.scatter(X, Y, s=1, c = sol_np1, cmap=cm.jet, vmin=-1, vmax=1.5)
        #plt.colorbar(sc)
        #plt.plot(np.cos(theta), np.sin(theta), 'k--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        #plt.savefig('plots/sol0p50_dt0p002.png')
        plt.show()
        
    if i == 374:
        
        plt.figure(figsize=(5,4))
        sc = plt.scatter(X, Y, s=1, c = sol_np1, cmap=cm.jet, vmin=-1, vmax=1.5)
        #plt.colorbar(sc)
        #plt.plot(np.cos(theta), np.sin(theta), 'k--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        #plt.savefig('plots/sol0p75_dt0p002.png')
        plt.show()
        
    if i == 499:
        
        plt.figure(figsize=(5,4))
        sc = plt.scatter(X, Y, s=1, c = sol_np1, cmap=cm.jet, vmin=-1, vmax=1.5)
        #plt.colorbar(sc)
        #plt.plot(np.cos(theta), np.sin(theta), 'k--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        #plt.savefig('plots/sol1p00_dt0p002.png')
        plt.show()

        
    return



# --- Single plot ----------------

# plt.figure(figsize=(5,4))
# sc = plt.scatter(X, Y, s=1, c = sol_np1, cmap=cm.jet, vmin=-1, vmax=1.5)
# plt.colorbar(sc)
# plt.plot(np.cos(theta), np.sin(theta), 'k--')
# plt.tight_layout()
# plt.savefig('sol0p25_dt0p002.pdf')
# plt.savefig('sol0p25_dt0p002.png')
# plt.show()
    

# ------ sub plot ----------------------    
    
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
# sc1 = ax1.scatter(X, Y, s=1, c = sol_np1, cmap=cm.jet, vmin=-1, vmax=1.5)
# sc2 = ax2.scatter(X, Y, s=1, c = init_sol, cmap=cm.jet, vmin=-1, vmax=1.5)

# ax1.plot(np.cos(theta), np.sin(theta), 'k--')
# ax2.plot(np.cos(theta), np.sin(theta), 'k--')
# plt.title('time = '+str(dt*(i+1)))
# plt.tight_layout()

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(sc2, cax=cbar_ax)

# plt.show()
"""


