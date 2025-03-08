import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def fixed_axis_fpf(yp, p_sect, p_ramp, p_slip, G=0.0):
    """
    fixed_axis_fpf plots the evolution of a simple step, 
    fixed axis fault propagation fold
    
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
    
    Based on the Matlab scripts FixedAxisFPF and
    FixedAxisFPFGrowth in Allmendinger et al. (2012)
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
    # first equation of Eq. 11.16
    gam_1 = (np.pi-ramp)/2.0
    tan_gam_1 = np.tan(gam_1)
    # second equation of Eq. 11.16
    gam_e_star = acot((3.0-2.0*cos_ramp)/(2.0*sin_ramp))
    # third equation of Eq. 11.16
    gam_i_star = gam_1 - gam_e_star
    # fourth equation of Eq. 11.16
    gam_e = acot(cot(gam_e_star)-2.0*cot(gam_1))
    tan_gam_e = np.tan(gam_e)
    cos_gam_e = np.cos(gam_e)
    sin_gam_e = np.sin(gam_e)
    # fifth equation of Eq. 11.16
    gam_i = np.arcsin((np.sin(gam_i_star)*np.sin(gam_e))/(np.sin(gam_e_star)))
    # ratio of backlimb length to total slip (P/S)(Eq. 11.17)
    a1 = cot(gam_e_star) - cot(gam_1)
    a2 = 1.0/sin_ramp - (np.sin(gam_i)/np.sin(gam_e))/(np.sin(gam_e+gam_i-ramp))
    a3 = np.sin(gam_1+ramp)/np.sin(gam_1)
    ps = a1/a2 + a3
    # change in slip between domains 2 and 3 (Eq. 11.19)
    R = np.sin(gam_1+ramp)/np.sin(gam_1+gam_e)
    # incremental crestal uplift of Hardy and Poblet (2005)
    inc_cr_upl = (np.sin(gam_1)/np.sin(gam_1+gam_e))*sin_gam_e

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

    # deform beds: Apply velocity fields of Eq. 11.18
    # in Allmendinger et al. (2012)
    # for first (1) to last (ninc) slip increment
    for i in range(1,ninc+1):
        # compute uplift
        lb = ps*i*sinc
        uplift = lb*sin_ramp
        lbh = lb*cos_ramp
        # compute fault tip
        xt = x_ramp + lbh
        yt = base + uplift
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
                    # if point is in domain 2
                    if XP[j,k] < xt - (YP[j,k] - yt)/tan_gam_1:
                        XP[j,k] += sinc*cos_ramp
                        YP[j,k] += sinc*sin_ramp
                    else:
                        # if point is in domain 3
                        if XP[j,k] < xt + (YP[j,k] - yt)/tan_gam_e:
                            XP[j,k] += sinc*R*cos_gam_e
                            YP[j,k] += sinc*R*sin_gam_e

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

def cot(x):
    """
    cotangent function
    """
    return 1.0/np.tan(x)

def acot(x):
    """
    inverse cotangent function
    """
    return np.arctan(1.0/x)