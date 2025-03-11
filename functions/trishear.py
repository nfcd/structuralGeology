import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def trishear(yp, p_sect, p_tri, sinc, G=0.0):
    """
    trishear plots the evolution of a 2D trishear 
    fault propagation fold

    Input:

    yp = n_beds-elements list with the elevation
        of the undeformed, horizontal beds
    p_sect = 2-elements list with the extent of the
        section, and the number of points in each bed
    p_tri = 7-elements list with the x coordinate of
        the fault tip, the y coordinate of the fault tip,
        the ramp angle, the P/S, the trishear angle,
        the fault slip, and the concentration factor
    sinc = slip increment
    G = subsidence versus uplift rate. Growth strata
        are added if G >= 1.0. The default is 0.0
        which means no growth strata are added.

    Note: ramp and trishear angles should be in radians
          For reverse faults use positive slip and increment
          For normal faults use negative slip and increment

    Returns: None but plots the evolution of the
        trishear fault propagation fold

    Based on the Matlab scripts Trishear, TrishearGrowth,
        and VelTrishear in Allmendinger et al. (2012)
    """
    # datums as an array of floats
    yp = np.array(yp).astype(float)
    
    # extent of section and number of points in each bed
    extent = p_sect[0]
    n_points = p_sect[1]
    
    # make undeformed beds geometry:
    # this is a grid of points along the beds
    point_int = extent / n_points
    xp = np.arange(0.0, extent+point_int, point_int)
    XP, YP = np.meshgrid(xp, yp)

    # trishear parameters
    xt = p_tri[0] # x at fault tip
    yt = p_tri[1] # y at fault tip
    ramp = p_tri[2] # ramp angle
    ps = p_tri[3] # P/S
    tra = p_tri[4] # trishear angle
    m = np.tan(tra/2) 
    slip = p_tri[5] # fault slip
    c = p_tri[6] # concentration factor

    # this speeds up the computation
    cos_ramp = np.cos(ramp)
    sin_ramp = np.sin(ramp)
    tan_ramp = np.tan(ramp)

    # number of slip increments
    ninc = round(slip / sinc)

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

    # tranformation matrix from geographic 
    # to fault coordinates
    a11 = cos_ramp
    a12 = np.cos(np.pi / 2 - ramp)
    a21 = np.cos(np.pi / 2 + ramp)
    a22 = a11

    # transform to coordinates parallel 
    # and perpendicular to the fault, and with
    # origin at initial fault tip
    FX = (XP - xt) * a11 + (YP - yt) * a12
    FY = (XP - xt) * a21 + (YP - yt) * a22

    # create a figure and axis
    fig, ax = plt.subplots()

    # loop over slip increments
    for i in range(1, ninc+1):
        # loop over number of beds
        for j in range(FX.shape[0]):
            # loop over number of points in each bed
            for k in range(FX.shape[1]):
                # solve trishear in a coordinate system 
                # attached to current fault tip (Eq. 11.27)
                xx = FX[j, k] - (ps * i * np.abs(sinc))
                yy = FY[j, k]
                # compute velocity (Eqs. 11.25 and 11.26)
                vx, vy = vel_trishear(xx, yy, sinc, m, c)
                # update coordinates
                FX[j, k] += vx
                FY[j, k] += vy

        # transform back to horizontal-vertical 
        # XP, YP coordinates for plotting
        XP = (FX * a11) + (FY * a21) + xt
        YP = (FX * a12) + (FY * a22) + yt

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
        ax.set_ylim(0, 2.0*max(yp))
        ax.set_aspect("equal")

        # plot beds, split hanging wall and footwall points
        for j in range(XP.shape[0]):
            c_hw = 0 # count hanging wall points
            for k in range(XP.shape[1]):
                # if hanging wall points
                if XP[j, k] <= xt + (YP[j, k] - yt) / tan_ramp:
                    c_hw += 1
            # pre-growth strata
            if j < yp.shape[0]:
                ax.plot(XP[j,:c_hw], YP[j,:c_hw], "k-")
                ax.plot(XP[j,c_hw:], YP[j,c_hw:], "k-")
            # growth strata
            else:
                ax.plot(XP[j,:c_hw], YP[j,:c_hw], "g-")
                ax.plot(XP[j,c_hw:], YP[j,c_hw:], "g-")

        # plot fault
        ax.plot(XF, YF, "r-", linewidth=2)

        # plot trishear boundaries
        ax.plot(XHTZ, YHTZ, "b-")
        ax.plot(XFTZ, YFTZ, "b-")

        # show amount of slip
        ax.text(0.8*extent, 1.75*max(yp), "Slip = " + str(i*np.abs(sinc)))

        # clear previous plot
        clear_output(wait=True)
        # redisplay the updated plot
        display(fig)

        # add growth strata
        if G >= 1.0:
            if i == count_g*ninc_g-1:
                # update top
                top += ninc_g * np.abs(sinc) * sin_ramp * G
                # make new bed
                xp = np.arange(i*sinc, extent+i*sinc+point_int, point_int)
                GXP,GYP = np.meshgrid(xp, [top])
                # transform to fault coordinates
                GFX = (GXP - xt) * a11 + (GYP - yt) * a12
                GFY = (GXP - xt) * a21 + (GYP - yt) * a22
                # add to existing beds
                FX = np.vstack((FX, GFX))
                FY = np.vstack((FY, GFY))
                # update count of growth layers
                count_g += 1

    # prevents Jupyter from displaying another plot
    plt.close(fig) 

def vel_trishear(xx, yy, sinc, m, c):
    """
    vel_trishear computes the velocity of a point
    in a trishear fault propagation fold. It is
    the symmetric, linear in vx trishear velocity field:
    Eq. 6 of Zehnder and Allmendinger (2000)

    Based on Matlab script VelTrishear 
    in Allmendinger et al. (2012)
    """
    # if behind the fault tip
    if xx < 0.0:
        # if hanging wall
        if yy >= 0.0:
            vx = sinc
            vy = 0.0
        # if footwall
        else:
            vx = 0.0
            vy = 0.0
    # if ahead the fault tip
    else:
        # if hanging wall
        if yy >= xx * m:
            vx = sinc
            vy = 0.0
        # if footwall
        elif yy <= -xx * m:
            vx = 0.0
            vy = 0.0
        # if inside the trishear zone
        else:
            a = 1.0 + c
            b = 1.0/c
            d = a/c
            ayy = np.abs(yy)
            syy = yy / ayy
            # Eq. 11.25
            vx = (sinc/2.0) * (syy * (ayy/(xx*m))**b + 1.0)
            # Eq. 11.26
            vy = (sinc/2.0) * (m/a) * ((ayy/(xx*m))**d - 1.0)
    
    return vx, vy

            


        





    


        
    