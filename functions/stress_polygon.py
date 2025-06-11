import numpy as np
import matplotlib.pyplot as plt

def stress_polygon(sv, pp, mu=0.6):
    """
    plots Zoback's stress polygons
    
    Input:
    sv: vertical stress in MPa
    pp: pore pressure in MPa
    mu: coefficient of friction (default 0.6)

    Returns: none but plots the stress polygons

    """
    # sigma_1 by sigma_3 ratio (Jaeger and Cook, 1979)
    s_rat = (np.sqrt(mu**2+1.0) + mu) ** 2
    # minimum horizontal stress, normal faulting
    # Eq. 4.45 in Zoback (2010)
    sh_min = pp + (sv - pp) / s_rat
    # maximum horizontal stress, reverse faulting
    # Eq. 4.47 in Zoback (2010)
    sh_max = pp + (sv - pp) * s_rat

    # create figure and set axes
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\sigma_{\mathrm{hmin}}$ (MPa)")
    ax.set_ylabel(r"$\sigma_{\mathrm{hmax}}$ (MPa)")
    ax.set_xlim(0, 3*sv)
    ax.set_ylim(0, 3*sv)
    ax.set_title(fr"$\sigma_v$ = {sv} MPa, $p$ = {pp} MPa")

    # unit slope line
    ax.plot([0, 3*sv], [0, 3*sv], "k")
    # normal faulting polygon
    nf_x = np.array([sh_min, sh_min, sv, sh_min])
    nf_y = np.array([sh_min, sv, sv, sh_min])
    ax.fill(nf_x, nf_y, "r", alpha=0.5, label="Normal")
    # strike-slip faulting polygon
    ss_x = np.array([sh_min, sv, sv, sh_min])
    ss_y = np.array([sv, sv, sh_max, sv])
    ax.fill(ss_x, ss_y, "g", alpha=0.5, label="Strike-slip")
    # reverse faulting polygon
    rf_x = np.array([sv, sv, sh_max, sv])
    rf_y = np.array([sv, sh_max, sh_max, sv])
    ax.fill(rf_x, rf_y, "b", alpha=0.5, label="Reverse")

    ax.legend()
    
    plt.show()

    