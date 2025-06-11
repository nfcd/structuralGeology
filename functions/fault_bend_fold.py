import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def fault_bend_fold(yp, p_sect, p_ramp, p_slip, G=0.0):
    """
    fault_bend_fold plots the evolution of a simple step, 
    Mode I fault bend fold
    
    Input:
    
    yp = n_beds-elements list with the elevation 
        of the undeformed, horizontal beds
    p_sect = 2-elements list with the extent of the 
        section, and the number of points in each bed 
    p_ramp = 3-elements list with the x coordinate 
        of the lower bend in the decollement, 
        the ramp angle, and the height of the ramp
    p_slip = 2-elements list with the total and 
        incremental slip
    G = subsidence versus uplift rate. Growth strata
        are added if G >= 1.0. The default is 0.0
        which means no growth strata are added.
 
    Note: Ramp angle should be in radians
    
    Returns: None but plots the evolution of the
        fault bend fold
    
    Based on the Matlab scripts FaultBendFold and
    FaultBendFoldGrowth in Allmendinger et al. (2012)
    """
    # extent of section and number of points in each bed
    extent = p_sect[0]
    n_points = p_sect[1]
    
    # make undeformed beds geometry: 
    # this is a grid of points along the beds
    point_int = extent / n_points
    xp = np.arange(0.0, extent+point_int, point_int)
    yp = np.array(yp).astype(float)
    XP, YP = np.meshgrid(xp, yp)

    # fault geometry and slip
    x_ramp = p_ramp[0]
    ramp = p_ramp[1]
    height = p_ramp[2]
    slip = p_slip[0]
    sinc = p_slip[1]

    # number of slip increments
    ninc = int(slip / sinc)

    # growth strata, if G >= 1.0
    if G >= 1.0:
        # top of pre-growth strata
        top = np.max(yp) 
        # number of growth layers
        n_g = 10 
        # number of slip increments per growth layer
        ninc_g = int(ninc/n_g)
        # initialize count of growth layers
        count_g = 1 

    # ramp angle cannot be greater than 30 degrees,
    # and if it is, make it 30 degrees
    if ramp > 30 * np.pi / 180:
        ramp = 30 * np.pi / 180
        print("Ramp angle was set to 30 degrees")
    
    # minimize Suppe's fault bend fold equation
    # to obtain gamma from the ramp angle
    gam_guess = 1.5
    gam = find_gam(gam_guess, ramp)

    # this makes the code faster
    tan_ramp = np.tan(ramp)
    tan_half_ramp = np.tan(ramp/2)
    cos_ramp = np.cos(ramp)
    sin_ramp = np.sin(ramp)
    tan_fg = np.tan(np.pi/2 - gam)
    
    # compute slip ratio R (Eq. 11.8)
    R = np.sin(gam - ramp) / np.sin(gam)

    # make arrays with the fault geometry
    xf = np.array([0, x_ramp, x_ramp + height/tan_ramp, 1.5*extent])
    yf = np.array([0, 0, height, height])

    # from the origin of each bed compute the number 
    # of points that are in the hanging wall;
    # these points are the ones that will move
    hw_points = np.zeros(yp.shape).astype(int)
    for i in range(yp.shape[0]):
        if yp[i] <= height:
            for j in range(xp.shape[0]):
                if xp[j] <= x_ramp + yp[i]/tan_ramp:
                    hw_points[i] += 1
        else:
            hw_points[i] = xp.shape[0]

    # the fold starts as stage 1
    stage = 1

    # create a figure and axis
    fig, ax = plt.subplots()
    
    # deform beds: Apply velocity fields of Eq. 11.9
    # in Allmendinger et al. (2012)
    # for first (1) to last (ninc) slip increment
    for i in range(1,ninc+1):
        # fold stage
        if stage == 1:
            if i*sinc*sin_ramp >= height:
                stage = 2
        # loop over number of beds
        for j in range(XP.shape[0]):
            # number of hanging wall points in bed
            # if pre-growth strata
            if j < yp.shape[0]:
                points = hw_points[j]
            # if growth strata
            else:
                points = XP.shape[1]
            # loop over number of hanging wall points in each bed
            for k in range(points):
                # if point is in domain 1
                if XP[j,k] < x_ramp - YP[j,k]*tan_half_ramp:
                    XP[j,k] += sinc
                else:
                    # if point is in domain 2
                    if YP[j,k] < height:
                        XP[j,k] += sinc*cos_ramp
                        YP[j,k] += sinc*sin_ramp
                    else:
                        # if stage 1 of fault bend fold 
                        # Fig. 11.3a of Allmendinger et al. (2012)
                        if i*sinc*sin_ramp < height:
                            # if point is in domain 2
                            if XP[j,k] < x_ramp + height/tan_ramp + (YP[j,k]-height)*tan_fg:
                                XP[j,k] += sinc*cos_ramp
                                YP[j,k] += sinc*sin_ramp
                            # if point is in domain 3
                            else:
                                XP[j,k] += sinc*R
                        # if stage 2 of fault bend fold (Fig. 11.3b)
                        else:
                            # if point is in domain 2
                            if XP[j,k] < x_ramp + height/tan_ramp - (YP[j,k]-height)*tan_half_ramp:
                                XP[j,k] += sinc*cos_ramp
                                YP[j,k] += sinc*sin_ramp
                            # if point is in domain 3
                            else:
                                XP[j,k] += sinc*R
    
        # clear axis
        ax.clear()
        
        # axis settings
        ax.set_xlim(0, extent)
        ax.set_ylim(0, 3.0*max(yp))
        ax.set_aspect("equal")
        
        # plot beds
        # pre-growth strata
        for j in range(yp.shape[0]):
            # if below ramp
            if yp[j] <= height:
                ax.plot(XP[j,:hw_points[j]],YP[j,:hw_points[j]],"k-")
                ax.plot(XP[j,hw_points[j]:],YP[j,hw_points[j]:],"k-")
            else:
                ax.plot(XP[j,:],YP[j,:],"k-")
        # growth strata
        for j in range(yp.shape[0], XP.shape[0]):
            ax.plot(XP[j,:],YP[j,:],"g-")

        # plot the fault
        ax.plot(xf,yf,"r-",linewidth=2)
        
        # show fold stage
        ax.text(0.8*extent, 2.8*max(yp), "Stage = " + str(stage))
        # show amount of slip
        ax.text(0.8*extent, 2.65*max(yp), "Slip = " + str(i*sinc))
        
        # clear previous plot
        clear_output(wait=True)
        # redisplay the updated plot
        display(fig)

        # add growth strata
        if G >= 1.0:
            if i == count_g*ninc_g-1:
                # update top
                tot_uplift = ninc_g * sinc * sin_ramp
                if tot_uplift < height:
                    top += tot_uplift * G
                # make new bed
                xp = np.arange(i*sinc, extent+i*sinc+point_int, point_int)
                GXP,GYP = np.meshgrid(xp, [top])
                # add to existing beds
                XP = np.vstack((XP, GXP))
                YP = np.vstack((YP, GYP))
                # update count of growth layers
                count_g += 1
            
    # prevents Jupyter from displaying another plot
    plt.close(fig)

def find_gam(gam_guess, ramp):
    """
    find the value of gamma that for a given ramp angle
    minimizes Suppe's fault bend fold equation
    Eq. 11.8 in Allmendinger et al. (2012)
    """
    def f(gam):
        return np.sin(2*gam) / (2*(np.cos(gam))**2 + 1) - np.tan(ramp)
    
    gam_sol = fsolve(f, gam_guess)
    
    return gam_sol[0]