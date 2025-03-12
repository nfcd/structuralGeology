import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from trishear import vel_trishear

def trishear_sed(yp, p_sect, p_tri, sed, diffusion=3.0, d_tz=False):
    """
    trishear_sed plots the tectonic and sedimentary
    evolution of a 2D trishear fault propagation fold.
    Background sedimentation and local erosion, transport
    and deposition as a result of trishear folding are
    considered.

    Input:

    yp = n_beds-elements list with the elevation (m)
        of the undeformed, horizontal beds
    p_sect = 2-elements list with the extent of the
        section (m), and the number of points in each bed
    p_tri = 6-elements list with the x coordinate of
        the fault tip, the y coordinate of the fault tip,
        the ramp angle, the P/S, the trishear angle,
        and the concentration factor
    sed = 5-elements list with the total run time (Ma), 
        base level rise (m/ka), background sedimentation 
        rate (m/ka), slip increment (m/ka), and time 
        interval at which growth layers are added (ka)
    diffusion = diffusion coefficient (m^2/a). The default
        is 3.0.
    d_tz = boolean. If True, the trishear boundaries are
        plotted. The default is False.

    Note: ramp and trishear angles should be in radians
          For reverse faults use positive slip and increment
          For normal faults use negative slip and increment
          Every increment of deformation is 1.0 ka

    Returns: None but plots the evolution of the
        trishear fault propagation fold

    Based on Hardy et al. (1996): Geological Society Spec.
    Publ. 99, 265-282.
    """
    # datums as an array of floats
    yp = np.array(yp).astype(float)
    # base level is the top of the pre-growth strata
    base_level = np.max(yp)
    # add another bed at the base level, 
    # this is the ground surface
    yp = np.append(yp, base_level)

    # extent of section and number of points in each bed
    extent = p_sect[0]
    n_points = p_sect[1]

    # make undeformed beds geometry:
    # this is a grid of points along the beds
    # notice that XP coordinates will not vary
    # since we will use a fixed x coordinate system
    point_int = extent / n_points
    xp = np.arange(0.0, extent+point_int, point_int)
    _, YP = np.meshgrid(xp, yp)

    # trishear parameters
    xt = p_tri[0] # x at fault tip
    yt = p_tri[1] # y at fault tip
    ramp = p_tri[2] # ramp angle
    ps = p_tri[3] # P/S
    tra = p_tri[4] # trishear angle
    m = np.tan(tra/2) 
    c = p_tri[5] # concentration factor

     # this speeds up the computation
    cos_ramp = np.cos(ramp)
    sin_ramp = np.sin(ramp)

    # total run time (Ma), base level rise (m/ka),
    # background sedimentation rate (m/ka), 
    # slip increment (m/ka), and time interval at which 
    # growth layers are added (ka)
    t_total = sed[0] * 1e3 # convert to ka
    base_level_rise = sed[1] 
    sed_rate = sed[2] 
    sinc = sed[3] 
    t_grow = sed[4]

    # diffusion in m^2/ka
    diffus = diffusion * 1e3 

    # number of slip increments
    ninc = int(t_total) 

    # tranformation matrix from geographic 
    # to fault coordinates
    a11 = cos_ramp
    a12 = np.cos(np.pi / 2 - ramp)
    a21 = np.cos(np.pi / 2 + ramp)
    a22 = a11

    # a time counter for growth layers
    c_grow = 0.0

    # smoothing factor for the ground surface
    # larger s means more smoothing
    # s = len(xp) seems to work well
    s = len(xp)

    # create a figure and axis
    fig, ax = plt.subplots()

    # loop over slip increments
    # each increment is t_inc ka
    for i in range(1, ninc+1):
        # base level
        base_level += base_level_rise
        # beds
        for j in range(YP.shape[0]):
            # if uppermost bed (ground surface)
            if j == YP.shape[0]-1:                
                # fit a smmooth  cubic spline to the ground surface
                spline = UnivariateSpline(xp, YP[j,:], s=s)
                # first derivative
                first_der = spline.derivative(1)(xp)
                # second derivative
                second_der = spline.derivative(2)(xp)
            # transform to fault coordinates
            f_xp = (xp - xt) * a11 + (YP[j,:] - yt) * a12
            f_yp = (xp - xt) * a21 + (YP[j,:] - yt) * a22
            # loop over points in bed
            for k in range(YP.shape[1]):
                # compute trishear deformation
                xx = f_xp[k] - (ps * i * np.abs(sinc))
                yy = f_yp[k]
                # compute velocity
                vx, vy = vel_trishear(xx, yy, sinc, m, c)
                # convert velocity to xp and yp coordinates
                u = vx * a11 + vy * a21
                v = vx * a12 + vy * a22
                # if uppermost bed (ground surface)
                if j == YP.shape[0]-1:
                    if YP[j,k] > base_level:
                        p = 0.0
                    else:
                        p = sed_rate
                    # Eq. 1 of Hardy et al. (1996)
                    d_v = (p + diffus * second_der[k]) + (v - u * first_der[k])
                # if pre-growth strata
                else:
                    d_v = v
                # update coordinates
                YP[j,k] += d_v
            
        # make fault geometry
        xtf = xt + (ps * i * np.abs(sinc)) * cos_ramp
        ytf = yt + (ps * i * np.abs(sinc)) * sin_ramp
        XF = np.array([xt, xtf])
        YF = np.array([yt, ytf])

        # make trishear boundaries
        axlo = np.arange(0, extent/3, extent/30)
        htz = axlo * m
        ftz = -axlo * m
        XHTZ = (axlo * a11 + htz * a21) + xtf
        YHTZ = (axlo * a12 + htz * a22) + ytf
        XFTZ = (axlo * a11 + ftz * a21) + xtf
        YFTZ = (axlo * a12 + ftz * a22) + ytf

        # clear previous plot
        ax.clear()
        
        # axis settings
        ax.set_xlim(0, extent)
        ax.set_ylim(0, 3.0*max(yp))
        ax.set_aspect("equal")

        # plot beds below ground surface
        for j in range(YP.shape[0]-1):
            # extract the points below the ground surface
            # which is the topmost bed
            x_b = xp[YP[j,:] <= YP[-1,:]]
            y_b = YP[j, YP[j,:] <= YP[-1,:]]
            # plot the points
            # pre-growth strata
            if j < yp.shape[0]-1:
                ax.plot(x_b, y_b, "k.", markersize=0.5)
            # growth strata
            else:
                ax.plot(x_b, y_b, "g.", markersize=0.5)

        # plot base level
        ax.plot([0, extent], [base_level, base_level], color="gray", linewidth=0.5)
        
        # plot uppermost bed (ground surface)
        ax.plot(xp, YP[-1,:], color="brown")

        # plot fault
        ax.plot(XF, YF, "r-", linewidth=2)

        # plot trishear boundaries
        if d_tz:
            ax.plot(XHTZ, YHTZ, "b-")
            ax.plot(XFTZ, YFTZ, "b-")
        
        # show amount of slip
        ax.text(0.75*extent, 2.75*max(yp), "Slip = " + str(i*np.abs(sinc)) + " m")

        # clear previous plot
        clear_output(wait=True)
        # redisplay the updated plot
        display(fig)

        # add growth layers
        c_grow += 1.0 # each increment is 1.0 ka
        if c_grow >= t_grow:
            # add growth layer = duplicate the topmost bed
            YP = np.vstack((YP, YP[-1,:]))
            # reset counter
            c_grow = 0.0

    # prevents Jupyter from displaying another plot
    plt.close(fig)

        