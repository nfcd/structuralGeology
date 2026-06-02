import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def similar_fold(yp, psect, fault, alpha, pslip, G=-1.0):
    """similar_fold plots the evolution of a similar fold

    USE: similar_fold(yp, psect, fault, alpha, pslip, G)

    yp    = Datums or vertical coordinates of undeformed, horizontal beds
    psect = A 1 x 2 vector containing the extent of the section, and the
            number of points in each bed
    fault = A npoints x 2 array containing the x and y coordinates
            of the fault
    alpha = Shear angle. Positive for shear antithetic to the fault and
            negative for shear synthetic to the fault
    pslip = A 1 x 2 vector containing the total and incremental slip
    G     = Ratio between regional subsidence and the local subsidence
            produced by the fault. G = 0.0 fills the accommodation up to 
            the top of the pre-growth; G = 1.0 regional subsidence is 
            equal to local subsidence, and so on. The default is -1.0, 
            which means no growth strata are added. 
            
            In case G >= 0, ten growth layers are added and drawn in green.

    NOTE: Use positive pslip for a normal fault

    Returns: None but plots the evolution of the similar fold

    Python translation of the MATLAB script SimilarFold by Nestor Cardozo
    (Structural Geology Algorithms, Allmendinger et al., 2012),
    extended with growth strata.
    """

    # Extent of section and number of points in each bed
    extent = psect[0]
    npoint = psect[1]

    # Make undeformed beds geometry: This is a grid of points along the beds
    xp = np.arange(0.0, extent + extent / npoint, extent / npoint)
    XP, YP = np.meshgrid(xp, yp)

    # Slip and number of slip increments
    slip = pslip[0]
    sinc = pslip[1]
    ninc = round(slip / sinc)

    # Sort fault points in x
    fault = np.asarray(fault, dtype=float)
    fault = fault[np.argsort(fault[:, 0])]
    xf = fault[:, 0]
    yf = fault[:, 1]
    n = fault.shape[0]

    # Find tangent of dip of fault segments: df/dx
    dfx = np.zeros(n)
    for i in range(n - 1):
        dfx[i] = (yf[i + 1] - yf[i]) / (xf[i + 1] - xf[i])
    dfx[n - 1] = dfx[n - 2]

    # Coordinate transformation matrix between horizontal-vertical coordinate
    # system and coordinate system parallel and perpendicular to shear direction
    a11 = np.cos(alpha)
    a12 = -np.sin(alpha)
    a21 = np.sin(alpha)
    a22 = a11

    # Number of pre-growth beds (used to colour growth strata differently)
    n_pre = np.size(yp)

    # From the origin of each bed compute the number of points that are in the
    # footwall.  These points won't move.  Stored in a list so growth beds
    # (added later) can append their own footwall counts.
    yfi = _interp1_extrap(xf, yf, xp)
    fwid = []
    for i in range(n_pre):
        cnt = 0
        for j in range(np.size(xp)):
            if yp[i] < yfi[j]:
                cnt += 1
        fwid.append(cnt)

    # Transform fault and beds to coordinate system parallel and perpendicular
    # to shear direction
    xfS = xf * a11 + yf * a12          # Fault
    XPS = XP * a11 + YP * a12          # Beds
    YPS = XP * a21 + YP * a22

    # --- growth-strata setup ---------------------------------------------- #
    # Growth strata are deposited whenever G >= 0 (i.e. always, unless the
    # caller sets G < 0 to switch growth off).  The depositional BASE LEVEL is
    # the top of the youngest pre-growth strata.  At G = 0 the base level stays
    # fixed there and growth fills only the accommodation created by the fault
    # (the subsiding rollover).  For G > 0 the base level also rises by G times
    # the local fault subsidence, adding a regional component on top.
    add_growth = G >= 0.0
    if add_growth:
        base = np.max(yp)             # base level = top of pre-growth strata
        n_g = 10                      # number of growth layers
        ninc_g = int(ninc / n_g)      # slip increments per growth layer
        count_g = 1                   # count of growth layers added so far
        # local fault subsidence per slip increment: vertical throw component
        # on the steepest part of the fault
        max_dip = np.arctan(np.max(np.abs(dfx)))
        sub_per_inc = np.abs(sinc) * np.sin(max_dip)

    # create a figure and axis
    fig, ax = plt.subplots()

    # Compute deformation
    # Loop over slip increments
    for i in range(1, ninc + 1):
        # Loop over number of beds (pre-growth + any growth beds added so far)
        for j in range(XPS.shape[0]):
            # Loop over number of bed points in hanging wall
            for k in range(fwid[j], XPS.shape[1]):
                # Find local tangent of fault dip: df/dx
                if XPS[j, k] <= xfS[0]:
                    ldfx = dfx[0]
                elif XPS[j, k] >= xfS[n - 1]:
                    ldfx = dfx[n - 1]
                else:
                    a = 'n'
                    L = 0
                    while a == 'n':
                        if XPS[j, k] >= xfS[L] and XPS[j, k] < xfS[L + 1]:
                            ldfx = dfx[L]
                            a = 's'
                        else:
                            L = L + 1
                # Compute velocities perpendicular and along shear direction
                # Equations 11.13 and 11.15
                vxS = sinc * a11
                vyS = (sinc * (a11 * a21 + ldfx * a11 ** 2)) / (a11 - ldfx * a21)
                # Move point
                XPS[j, k] = XPS[j, k] + vxS
                YPS[j, k] = YPS[j, k] + vyS

        # Transform beds back to geographic coordinate system
        XP = XPS * a11 + YPS * a21
        YP = XPS * a12 + YPS * a22

        # clear previous plot
        ax.clear()

        # axis settings
        ax.set_xlim(0, extent)
        ax.set_ylim(0, 2.0 * np.max(yp))
        ax.set_aspect("equal")

        # Beds
        for j in range(XP.shape[0]):
            color = 'k-' if j < n_pre else 'g-'      # growth strata in green
            # Footwall
            ax.plot(XP[j, 0:fwid[j]], YP[j, 0:fwid[j]], color)
            # Hanging wall
            ax.plot(XP[j, fwid[j]:XP.shape[1]],
                    YP[j, fwid[j]:XP.shape[1]], color)

        # Fault: when growth strata have raised the base level above the fault's
        # uppermost point, extend the fault up to the current growth surface
        # along its uppermost dip.
        xf_plot, yf_plot = xf, yf
        if add_growth and base > np.max(yf):
            ytop = np.max(yf)                       # shallowest fault point
            ktop = int(np.argmax(yf))
            x_top = xf[ktop]
            dip_top = dfx[ktop]                     # dy/dx of the top segment
            if dip_top != 0.0:
                x_ext = x_top + (base - ytop) / dip_top   # x where fault hits base
            else:
                x_ext = x_top
            xf_plot = np.concatenate([[x_ext], xf])
            yf_plot = np.concatenate([[base], yf])
        ax.plot(xf_plot, yf_plot, 'r-', linewidth=2)

        # show amount of slip
        ax.text(0.8 * extent, 1.75 * np.max(yp),
                'Slip = ' + str(i * sinc))

        # clear previous plot
        clear_output(wait=True)
        # redisplay the updated plot
        display(fig)

        # --- add a growth layer every ninc_g increments -------------------- #
        if add_growth and i == count_g * ninc_g - 1:
            # base level = top of pre-growth strata, raised by the REGIONAL
            # subsidence over this interval = G times the local fault
            # subsidence.  G = 0 keeps base level fixed; growth then fills only
            # the fault-generated accommodation up to the pre-growth top.
            base = base + G * ninc_g * sub_per_inc
            # new horizontal bed of points spanning the section, at base level
            GXP, GYP = np.meshgrid(xp, np.array([base], dtype=float))
            # footwall count for the new bed (points where bed is below fault)
            cnt = 0
            for jj in range(np.size(xp)):
                if base < yfi[jj]:
                    cnt += 1
            fwid.append(cnt)
            # transform the new bed into shear coordinates and append
            GXPS = GXP * a11 + GYP * a12
            GYPS = GXP * a21 + GYP * a22
            XPS = np.vstack((XPS, GXPS))
            YPS = np.vstack((YPS, GYPS))
            count_g += 1

    # prevents Jupyter from displaying another plot
    plt.close(fig)


def _interp1_extrap(xf, yf, xp):
    """Equivalent of MATLAB interp1(xf, yf, xp, 'linear', 'extrap')."""
    yfi = np.interp(xp, xf, yf)
    if len(xf) > 1:
        mL = (yf[1] - yf[0]) / (xf[1] - xf[0])
        left = xp < xf[0]
        yfi[left] = yf[0] + mL * (xp[left] - xf[0])
        mR = (yf[-1] - yf[-2]) / (xf[-1] - xf[-2])
        right = xp > xf[-1]
        yfi[right] = yf[-1] + mR * (xp[right] - xf[-1])
    return yfi


def listric_fault(upper, lower, npts=40, x_end=None):
    """Build a listric fault geometry as an (npts x 2) array of (x, y) points.

    A smooth cubic Hermite curve is fitted between an upper and a lower point,
    honouring the prescribed dip (tangent) at each end, giving a fault that is
    steep at the top and flattens with depth (or whatever the two dips imply).

    Parameters
    ----------
    upper : (x, y, dip) of the uppermost fault point.
    lower : (x, y, dip) of the lowermost fault point.
            dip is in RADIANS, positive to the right (i.e. the fault goes DOWN
            to the right, so dy/dx = -tan(dip)).
    npts  : number of points used to sample the Hermite curve.
    x_end : if given and > lower x, the fault is extended horizontally (flat)
            from the lower point out to x = x_end (e.g. a basal detachment).

    Returns
    -------
    (N x 2) ndarray of (x, y) fault points, sorted in x.
    """
    xu, yu, dipu = upper
    xl, yl, dipl = lower
    mu = -np.tan(dipu)              # dy/dx at the upper point
    ml = -np.tan(dipl)              # dy/dx at the lower point

    # order the two control points by x so the Hermite parameter runs forward
    if xl < xu:
        (x0, y0, m0), (x1, y1, m1) = (xl, yl, ml), (xu, yu, mu)
    else:
        (x0, y0, m0), (x1, y1, m1) = (xu, yu, mu), (xl, yl, ml)

    L = x1 - x0
    t = np.linspace(0.0, 1.0, npts)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    xc = x0 + t * L
    yc = h00 * y0 + h10 * L * m0 + h01 * y1 + h11 * L * m1

    pts = np.column_stack([xc, yc])

    # optional flat extension beyond the lower point (the detachment)
    x_low = max(xu, xl)            # the lower point is the more basinward one
    if x_end is not None and x_end > x_low:
        y_low = yl if xl >= xu else yu
        xext = np.linspace(x_low, x_end, max(2, int((x_end - x_low) /
                                                    (L / npts)) + 1))[1:]
        pts = np.vstack([pts, np.column_stack([xext, np.full_like(xext, y_low)])])

    return pts[np.argsort(pts[:, 0])]
