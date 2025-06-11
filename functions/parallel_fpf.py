import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def parallel_fpf(yp, p_sect, p_ramp, p_slip, G=0.0):
    """
    parallel_fpf plots the evolution of a simple step, 
    parallel fault propagation fold
    
    Input:
    
    yp = n_beds-elements list with the elevation 
        of the undeformed, horizontal beds
    p_sect = 2-elements list with the extent of the 
        section, and the number of points in each bed 
    p_ramp = 2-elements list with the x coordinate 
        of the lower bend in the decollement, 
        and the ramp angle
    p_slip = 2-elements list with the total and 
        incremental slip
    G = subsidence versus uplift rate. Growth strata
        are added if G >= 1.0. The default is 0.0
        which means no growth strata are added.
 
    Note: Ramp angle should be in radians
    
    Returns: None but plots the evolution of the 
        fault propagation fold
    
    Based on the Matlab scripts ParallelFPF and
    ParallelFPFGrowth in Allmendinger et al. (2012)
    """
    # datums as an array of floats
    yp = np.array(yp).astype(float)
    
    # Base of layers
    base=yp[0]

    # extent of section and number of points in each bed
    extent = p_sect[0]
    n_points = p_sect[1]

    # make undeformed beds geometry: 
    # this is a grid of points along the beds
    point_int = extent / n_points
    xp = np.arange(0.0, extent+point_int, point_int)
    XP, YP = np.meshgrid(xp, yp)

    # fault geometry and slip
    x_ramp = p_ramp[0]
    ramp = p_ramp[1]
    slip = p_slip[0]
    sinc = p_slip[1]

    # this saves time
    tan_ramp = np.tan(ramp)
    sin_ramp = np.sin(ramp)
    cos_ramp = np.cos(ramp)

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

    # solve model parameters
    # Eqs. from Allmendinger et al. (2012)
    # first equation of Eq. 11.20
    gam_guess = 0.5
    gam_star = find_gam_star(gam_guess, ramp)
    # second equation of Eq. 11.20
    gam_1 = np.pi/2.0 -ramp/2.0
    tan_gam_1 = np.tan(gam_1)
    # third equation of Eq. 11.20
    gam = np.pi/2.0 + gam_star - gam_1
    cos_gam = np.cos(gam)
    sin_gam = np.sin(gam)
    tan_gam = np.tan(gam)
    # fourth equation of Eq. 11.20
    bet_2 = np.pi - 2.0*gam_star
    # another angle for calculations
    kap = np.pi - bet_2 + ramp
    cos_kap = np.cos(kap)
    sin_kap = np.sin(kap)
    tan_kap = np.tan(kap)
    # Eq. 11.21
    ps = 1.0/(1.0-sin_ramp/np.sin(2.0*gam-ramp))
    # Eq. 11.23
    R1 = np.sin(gam_1+ramp)/np.sin(gam_1+gam)
    R2 = np.sin(bet_2)/np.sin(bet_2-ramp+gam)
    # incremental Crestal uplift. Eq. 15 of Hardy and Poblet (2005)
    inc_cr_upl = (np.sin(gam_1)/np.sin(gam_1+gam))*sin_gam

    # from the origin of each bed compute the number 
    # of points that are in the hanging wall;
    # these points are the ones that will move. 
    # Notice that since the fault propagates, this 
    # should be done for each slip increment.
    hw_points = np.zeros((ninc, yp.shape[0])).astype(int)
    for i in range(1,ninc+1):
        uplift = ps*i*sinc*sin_ramp
        for j in range(yp.shape[0]):
            if yp[j] - base <= uplift:
                hw_points[i-1,j] = 0
                for k in range(xp.shape[0]):
                    if xp[k] <= x_ramp + (yp[j] - base)/tan_ramp:
                        hw_points[i-1,j] += 1
            else:
                hw_points[i-1,j] = xp.shape[0]

    # create a figure and axis
    fig, ax = plt.subplots()

    # deform beds: Apply velocity fields of Eq. 11.22
    # in Allmendinger et al. (2012)
    # for first (1) to last (ninc) slip increment
    for i in range(1,ninc+1):
        # compute uplift
        lb = ps*i*sinc
        uplift = lb*sin_ramp
        lbh = lb*cos_ramp
        # compute distance ef in Fig. 11.6 
        # of Allmendinger et al. (2012)
        ef = uplift/np.sin(2.0*gam_star)
        # compute fault tip
        xt = x_ramp + lbh
        yt = base + uplift
        # compute location e in Fig. 11.6
        xe = xt + ef*cos_kap
        ye = yt + ef*sin_kap
        # loop over number of beds
        for j in range(XP.shape[0]):
            # number of hanging wall points in bed
            # if pre-growth strata
            if j < yp.shape[0]:
                points = hw_points[i-1,j]
            # if growth strata
            else:
                points = XP.shape[1]
            # loop over number of hanging wall points in each bed
            for k in range(points):
                # if point is in domain 1
                if XP[j,k] < x_ramp - (YP[j,k] - base)/tan_gam_1:
                    XP[j,k] += sinc
                else:
                    # if lower than location e (Fig. 11.6)
                    if YP[j,k] < ye:
                        # if point is in domain 2
                        if XP[j,k] < xt + (YP[j,k] - yt)/tan_kap:
                            XP[j,k] += sinc*cos_ramp
                            YP[j,k] += sinc*sin_ramp
                        else:
                            # if point is in domain 4
                            if XP[j,k] < xt + (YP[j,k] - yt)/tan_gam:
                                XP[j,k] += sinc*R2*cos_gam
                                YP[j,k] += sinc*R2*sin_gam

                    # if higher than location e (Fig. 11.6)
                    else:
                        # if point is in domain 2
                        if XP[j,k] < xe - (YP[j,k] - ye)/tan_gam_1:
                            XP[j,k] += sinc*cos_ramp
                            YP[j,k] += sinc*sin_ramp
                        else:
                            # if point is in domain 3
                            if XP[j,k] < xe + (YP[j,k] - ye)/tan_gam:
                                XP[j,k] += sinc*R1*cos_gam
                                YP[j,k] += sinc*R1*sin_gam
                            else:
                                # if point is in domain 4
                                if XP[j,k] < xt + (YP[j,k] - yt)/tan_gam:
                                    XP[j,k] += sinc*R2*cos_gam
                                    YP[j,k] += sinc*R2*sin_gam

        # clear axis
        ax.clear()

        # axis settings
        ax.set_xlim(0, extent)
        ax.set_ylim(0, 3.0*max(yp))
        ax.set_aspect("equal")

        # plot beds
        # pre-growth strata
        for j in range(yp.shape[0]):
            # if beds cut by the fault
            if yp[j] - base <= uplift:
                ax.plot(XP[j,:hw_points[i-1,j]], YP[j,:hw_points[i-1,j]], "k-")
                ax.plot(XP[j,hw_points[i-1,j]:], YP[j,hw_points[i-1,j]:], "k-")
            # if beds not cut by the fault
            else:
                ax.plot(XP[j,:], YP[j,:], "k-")
        # growth strata
        for j in range(yp.shape[0], XP.shape[0]):
            ax.plot(XP[j,:],YP[j,:],"g-")

        # plot fault
        xf = np.array([0, x_ramp, x_ramp + lbh])
        yf = np.array([base, base, uplift+base])
        ax.plot(xf, yf, "r-", linewidth=2)
        
        # show amount of slip
        ax.text(0.8*extent, 2.75*max(yp), "Slip = " + str(i*sinc))

        # clear previous plot
        clear_output(wait=True)
        # redisplay the updated plot
        display(fig)

        # add growth strata
        if G >= 1.0:
            if i == count_g*ninc_g-1:
                # update top
                top += ninc_g*sinc*inc_cr_upl*G
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

def find_gam_star(gam_guess, ramp):
    """
    find the value of gamma star that for a given ramp angle
    minimizes Suppe and Medwedeff's fault propagation fold 
    equation. Eq. 11.20 in Allmendinger et al. (2012)
    """
    def f(gam_star):
        return (1.+2.*np.cos(gam_star)**2)/np.sin(2.*gam_star) + (np.cos(ramp)-2.)/np.sin(ramp)
    
    gam_sol = fsolve(f, gam_guess)
    
    return gam_sol[0]