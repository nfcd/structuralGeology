import numpy as np 
import matplotlib.pyplot as plt 

def hafner(a, b, c, theta, sigma_0, n, L, D, grid_x, grid_y):
    """
    Computes the 2D stress distribution based on Hafner (1951):
    Stress distribution and faulting, GSA Bull. v. 62, 373-398.
    
    Parameters:
    a : float
        Vertical gradient due to weight (MPa/km)
    b : float
        Vertical gradient of superimposed horizontal pressure (MPa/km)
    c : float
        Lateral gradient of superimposed horizontal pressure (MPa/km)
    theta : float
        Angle between fault plane and maximum compressive stress (radians)
    sigma_0 : float
        Unconfined compressive strength of the rock (MPa)
    n : float
        Coefficient for the increase of strength with sigma_max
    L : float
        Horizontal extent (km)
    D : float
        Vertical extent (km)
    grid_x : int
        Number of points in the horizontal dimension for the grid
    grid_y : int
        Number of points in the vertical dimension for the grid
    
    Returns:
    fig, ax : matplotlib figure and axes objects
        The upper figure shows the principal stress directions 
        and contours of maximum shear stress.
        The lower figure shows the fault trajectories and
        stable (no faulting) area in white.

    Note: Compressional stresses are considered negative, and
            tensile stresses positive.
    """
    # Create grid, for nice plotting extend beyond L and D
    x = np.linspace(-2*L, 0, grid_x*2)
    y = np.linspace(0, 2*D, grid_y*2)
    X, Y = np.meshgrid(x, y)
    
    # Stress components, Hafner (1951) 1st case, equations in Fig. 7 
    sigma_x = c * X - b * Y - a * Y 
    sigma_y = -a * Y 
    tau_xy = -c * Y 
    
    # Maximum shear stress, Hafner (1951) Eq. (3) 
    diff_sigma = sigma_x - sigma_y 
    tau_max = np.sqrt((diff_sigma / 2)**2 + tau_xy**2)

    # Principal stress magnitudes, Hafner (1951) Eq. (4)
    avg = (sigma_x + sigma_y) * 0.5 
    s_min = avg - tau_max # maximum compressive stress
    s_max = avg + tau_max # minimum compressive stress

    # Principal stress directions, Hafner (1951) Eq. (5)
    beta = 0.5 * np.arctan2(2 * tau_xy, diff_sigma)
    u_max, v_max = np.cos(beta), np.sin(beta) # sigma_max 
    u_min, v_min = np.cos(beta + np.pi/2), np.sin(beta + np.pi/2) # sigma_min

    # direction vectors for fault trajectories: +/- theta to sigma_min 
    u_f1, v_f1 = np.cos(beta+np.pi/2+theta), np.sin(beta+np.pi/2+theta) 
    u_f2, v_f2 = np.cos(beta+np.pi/2-theta), np.sin(beta+np.pi/2-theta)

    # stability criterion, Hafner (1951) Eq. (8)
    strength = n * s_max + sigma_0 
    stability = s_min - strength

    # make a figure with two subplots 
    fig, ax = plt.subplots(2, 1, figsize=(12, 8)) 

    # plot maximum shear stress contours
    mask = (X >= -L) & (X <= 0) & (Y >= 0) & (Y <= D)
    tau_max_masked = np.where(mask, tau_max, np.nan)   
    levels = np.arange(c, 10*c, c) 
    contours = ax[0].contour(X, Y, tau_max_masked, levels=levels, 
                             colors='gray', linewidths=0.8) 
    ax[0].clabel(contours, fmt={l: f'{-int(l/c)}c' for l in levels}, 
                 inline=True, fontsize=10) 
    
    # plot principal stress trajectories 
    # s_min: (solid lines) 
    ax[0].streamplot(X, Y, u_min, v_min, color='black', linewidth=0.9, 
                     arrowstyle='-', density=1, minlength=0.9) 
    # s_max: (dashed lines) 
    strm2 = ax[0].streamplot(X, Y, u_max, v_max, color='black', linewidth=0.8, 
                         arrowstyle='-', density=2, minlength=0.9) 
    strm2.lines.set_linestyle((0, (5, 5))) 

    # plot fault trajectories 
    ax[1].streamplot(X, Y, u_f1, v_f1, color='black', linewidth=0.9, 
                     arrowstyle='-', density=1, minlength=0.9) 
    ax[1].streamplot(X, Y, u_f2, v_f2, color='black', linewidth=0.9, 
                     arrowstyle='-', density=1, minlength=0.9) 
    
    # plot stable area 
    contours = ax[1].contour(X, Y, stability, levels=[0], colors='black', 
              linewidths=1.5, zorder=3) 
    contours.set_linestyle((0, (5, 3, 1, 5)))
    ax[1].contourf(X, Y, stability, levels=[0, 1000], colors='white', zorder=2) 

    # format axes
    for ax_i in ax: 
        ax_i.set_xlim(-L, -1) 
        ax_i.set_ylim(D, 0) 
        ax_i.set_aspect('equal') 
        ax_i.set_xlabel("x [km]") 
        ax_i.set_ylabel("y [km]") 

    fig.tight_layout() 
    plt.show() 

    return fig, ax
    