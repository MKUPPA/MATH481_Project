"""
Author: Mridula Kuppa

Description: Utilities to plot solution post heat equation solution on unit sphere

"""


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

    # Plot solution on sphere
    
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
