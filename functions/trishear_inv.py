import itertools
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from IPython.display import display, clear_output
from trishear import vel_trishear

def back_trishear(xp, yp, tpar, sinc):
    """
    Backward trishear deformation.
    
    Parameters
    ----------
    xp: array-like x-coordinates of points along the bed.
    yp: array-like y-coordinates of points along the bed.
    tpar: 7-element list with trishear parameters:
        xtf : x-coordinate of the fault tip.
        ytf : y-coordinate of the fault tip.
        ramp : ramp angle in radians.
        ps : P/S ratio.
        tra : trishear angle in radians.
        slip : fault slip.
        c : concentration factor.
    sinc: slip increment.

    Returns
    -------
    chisq = sum of squared residuals between the restored 
            bed and a linear fit to it.

    Based on the Matlab scripts BackTrishear,
    in Allmendinger et al. (2012)
    """
    # model parameters
    xtf = tpar[0] # x at fault tip
    ytf = tpar[1] # y at fault tip
    ramp = tpar[2] # ramp angle
    psr = tpar[3] * -1.0 # P/S: restoring bed
    tra = tpar[4] # trishear angle
    m = np.tan(tra/2)
    slip = tpar[5] # fault slip
    c = tpar[6] # concentration factor
    ninc = round(slip / sinc) # number of slip increments
    sincr = slip/ninc * -1.0 # slip increment: restoring bed

    # transformation matrix from geographic to fault coordinates
    a11 = np.cos(ramp)
    a12 = np.cos(np.pi / 2 - ramp)
    a21 = np.cos(np.pi / 2 + ramp)
    a22 = a11

    # transform to coordinates parallel and perpendicular 
    # to the fault, and with origin at fault tip
    fx = (xp - xtf) * a11 + (yp - ytf) * a12
    fy = (xp - xtf) * a21 + (yp - ytf) * a22

    # restore the bed
    for i in range(ninc):
        for j in range(len(fx)):
            # solve trishear in a coordinate system
            # attached to the fault tip. Note: first
            # retrodeform and then move tip back.
            xx = fx[j] - (psr * i * np.abs(sincr))
            yy = fy[j]
            # compute trishear velocity
            vx, vy = vel_trishear(xx, yy, sincr, m, c)
            # update fx and fy
            fx[j] = fx[j] + vx
            fy[j] = fy[j] + vy

    # fit a line to the retrodeformed bed
    A = np.vstack([fx, np.ones(len(fx))]).T
    result = np.linalg.lstsq(A, fy, rcond=None)

    # compute chisq: sum of squared residuals
    if len(result[1]) > 0:
        chisq = result[1][0]
    else:
        yavg = np.mean(fy)
        chisq = np.sum((fy - yavg)**2)

    return chisq

def grid_search(xp, yp, bounds, sinc, output=False):
    """
    Grid search for the best trishear parameters.
    
    Parameters
    ----------
    xp: array-like x-coordinates of points along the bed.
    yp: array-like y-coordinates of points along the bed.
    bounds : 7-element list where each element is 
        [min, max, step] of the trishear parameter.
        If a parameter is fixed, min == max.
        Order: xtf, ytf, ramp, ps, tra, slip, c
    sinc: slip increment.
    output: if True, print iteration log.

    Returns
    -------
    tpar_best: list of best-fit trishear parameters.
    chisq_min: minimum chisq value.
    chisq: array of chisq values for each parameter combination.
    """
    # check that bounds have 7 elements
    if len(bounds) != 7:
        raise ValueError("bounds must have 7 elements")
    
    # check that bounds are correctly formatted
    for i, bound in enumerate(bounds):
        if len(bound) != 3:
            raise ValueError("each bound must have 3 elements")
        if bound[0] > bound[1]:
            raise ValueError("bounds must be in the order [min, max, step]")
        if bound[2] < 0 and bound[0] < bound[1]:
            raise ValueError(f"step size must be positive")
    
    # initialize trishear parameters to the minimum values
    tpar = [b[0] for b in bounds]
    
    # find the free parameters
    index_free = []
    value_ranges = []
    for i, (low, high, step) in enumerate(bounds):
        if low < high:
            index_free.append(i)
            nels = int(round((high - low) / step)) + 1
            value_ranges.append(np.linspace(low, high, nels))
    
    # prepare chisq array with shape based on number of steps per param
    shape = [len(r) for r in value_ranges]
    chisq = np.full(shape, np.inf)

    # for index mapping
    index_ranges = [range(len(r)) for r in value_ranges]

    # initialize tpar_best, chisq_min and count
    tpar_best = tpar.copy()
    chisq_min = np.inf
    count = 0  # number of iterations

    if output:
        # total number of models
        total_models = np.prod(shape)
        print(f"Total models: {total_models}")
        # headings for the output
        print("Model, [xt, yt, ramp, ps, tra, slip, c], chisq")

    # iterate over all combinations of the free parameters
    for idx_tuple in itertools.product(*index_ranges):
        # update free parameters in tpar
        for i, param_index in enumerate(index_free):
            tpar[param_index] = value_ranges[i][idx_tuple[i]]

        # compute chisq
        chisq_temp = back_trishear(xp, yp, tpar, sinc)

        # store chisq
        chisq[idx_tuple] = chisq_temp

        # check for best result
        if chisq_temp < chisq_min:
            chisq_min = chisq_temp
            tpar_best = tpar.copy()

        if output:
            count += 1
            par_str = [f"{x:.2f}" for x in tpar]
            print(count, par_str, f"{chisq_temp:.2f}")

    return tpar_best, chisq_min, chisq      

    
    
def simulated_annealing(xp, yp, bounds, sinc, maxiter=100, 
                        initial_temp=5230.0, seed=None, 
                        output=False):
    """
    Simulated annealing for the best trishear parameters

    Parameters
    ----------
    xp : array-like x-coordinates of points along the bed.
    yp : array-like y-coordinates of points along the bed.
    bounds : 7-element list where each element is 
        [min, max] of the trishear parameter.
        If a parameter is fixed, min == max.
        Order: xtf, ytf, ramp, ps, tra, slip, c
    sinc : slip increment.
    maxiter : optional, maximum number of cycles.
    initial_temp : optional, initial temperature for 
                   dual_annealing.
    seed : optional, random seed for reproducibility.
    output : optional, if True, print progress.

    Returns
    -------
    tpar_best : list of best trishear parameters
    chisq_min : minimum chisq value found
    history : array with rows [parameters..., chisq] for 
              each evaluation.
    """
    # Validate bounds
    if len(bounds) != 7:
        raise ValueError("bounds must have 7 elements")
    for b in bounds:
        if len(b) != 2:
            raise ValueError("Each bound must have two elements [min, max].")
        if b[0] > b[1]:
            raise ValueError("Bounds must be in the order [min, max].")

    # Initial parameters
    tpar = [b[0] for b in bounds]

    # Identify free parameters and their bounds
    index_free = [i for i, b in enumerate(bounds) if b[0] < b[1]]
    bounds_free = [bounds[i] for i in index_free]

    history = []

    if output:
        print("Model, [xt, yt, ramp, ps, tra, slip, c], chisq")

    # Objective function to minimize
    def objective(params):
        # update tpar with free parameters
        for idx, val in zip(index_free, params):
            tpar[idx] = val

        chisq = back_trishear(xp, yp, tpar, sinc)
        history.append((tpar.copy(), chisq))

        if output:
            par_str = [f"{x:.2f}" for x in tpar]
            print(len(history), par_str, f"{chisq:.2f}")

        return chisq

    # Prepare random seed dict for dual_annealing 
    # (only available in recent scipy versions)
    seed_kwargs = {"seed": seed} if seed is not None else {}

    # Run dual_annealing optimizer
    result = dual_annealing(objective, bounds_free,
                            maxiter=maxiter,
                            initial_temp=initial_temp,
                            no_local_search=True,
                            **seed_kwargs)

    # Extract best parameters into full vector
    tpar_best = tpar.copy()
    for idx, val in zip(index_free, result.x):
        tpar_best[idx] = val

    chisq_min = result.fun

    # Convert history to numpy array with shape (num_iters, 8)
    # Each row: 7 parameters + chisq
    tpar_history = np.array([h[0] for h in history])
    chisq_history = np.array([h[1] for h in history])
    history = np.column_stack((tpar_history, chisq_history))

    return tpar_best, chisq_min, history

def restore_beds(beds, tpar, sinc):
    """
    Restore the beds using trishear parameters.
    
    Parameters
    ----------
    beds: list of beds to restore.
    tpar: 7-element list with trishear parameters
          Order: xtf, ytf, ramp, ps, tra, slip, c
    sinc: slip increment.

    Returns
    -------
    beds_rest: list of linear fits to the restored beds.
    xti: x-coordinate of the initial fault tip.
    yti: y-coordinate of the initial fault tip.
    """
    # minimum and maximum x and y-coordinates of the beds
    x_min = min([min(bed[:, 0]) for bed in beds])
    x_max = max([max(bed[:, 0]) for bed in beds])
    y_min = min([min(bed[:, 1]) for bed in beds])
    y_max = max([max(bed[:, 1]) for bed in beds])
    # extent of section
    extent_x = x_max - x_min
    extent_y = y_max - y_min   
    
    # model parameters
    xtf = tpar[0] # x at fault tip
    ytf = tpar[1] # y at fault tip
    ramp = tpar[2] # ramp angle
    psr = tpar[3] * -1.0 # P/S: restoring bed
    tra = tpar[4] # trishear angle
    m = np.tan(tra/2)
    slip = tpar[5] # fault slip
    c = tpar[6] # concentration factor
    ninc = round(slip / sinc) # number of slip increments
    sincr = slip/ninc * -1.0 # slip increment: restoring bed

    # transformation matrix from geographic to fault coordinates
    a11 = np.cos(ramp)
    a12 = np.cos(np.pi / 2 - ramp)
    a21 = np.cos(np.pi / 2 + ramp)
    a22 = a11

    # restored beds
    beds_rest = copy.deepcopy(beds)

    # beds transformed to fault coordinates and with origin at fault tip
    beds_transf = []
    for bed in beds_rest:
        fx = (bed[:, 0] - xtf) * a11 + (bed[:, 1] - ytf) * a12
        fy = (bed[:, 0] - xtf) * a21 + (bed[:, 1] - ytf) * a22
        beds_transf.append(np.column_stack((fx, fy)))

    # create a figure and axis
    fig, ax = plt.subplots()

    # restore the beds
    for i in range(ninc):
        for j in range(len(beds_transf)):
            fx, fy = beds_transf[j][:, 0], beds_transf[j][:, 1]
            for k in range(len(fx)):
                # solve trishear in a coordinate system
                # attached to the fault tip. Note: first
                # retrodeform and then move tip back.
                xx = fx[k] - (psr * i * np.abs(sincr))
                yy = fy[k]
                # compute trishear velocity
                vx, vy = vel_trishear(xx, yy, sincr, m, c)
                # update fx and fy
                fx[k] = fx[k] + vx
                fy[k] = fy[k] + vy
            # update the transformed bed
            beds_transf[j][:, 0] = fx
            beds_transf[j][:, 1] = fy
            # update the restored bed
            beds_rest[j][:, 0] = (beds_transf[j][:, 0] 
                                  * a11 + beds_transf[j][:, 1] * a21) + xtf
            beds_rest[j][:, 1] = (beds_transf[j][:, 0] 
                                  * a12 + beds_transf[j][:, 1] * a22) + ytf
        # restored fault tip
        xti = xtf + (psr * i * np.abs(sincr)) * a11
        yti = ytf + (psr * i * np.abs(sincr)) * a12

        # clear previous plot
        ax.clear()
        
        # axis settings
        ax.set_xlim(x_min-extent_x*1.5, x_max+extent_x*0.1)
        ax.set_ylim(y_min-extent_y*1.5, y_max+extent_y*0.1)
        ax.set_aspect("equal")

        # plot the deformed beds in gray
        for bed in beds:
            ax.plot(bed[:, 0], bed[:, 1], ".", color="gray", markersize=1)
        # plot the restored beds in blue
        for bed in beds_rest:
            ax.plot(bed[:, 0], bed[:, 1], "b.", markersize=1)
        # plot the fault
        ax.plot([xti, xti-125*a11], [yti, yti-125*a12], "r-", linewidth=2)
        ax.plot(xti, yti, "ro", markersize=5)

        # clear previous plot
        clear_output(wait=True)
        # redisplay the updated plot
        display(fig)

    # prevents Jupyter from displaying another plot
    plt.close(fig)

    # lines fits to the restored beds
    for i in range(len(beds_rest)):
        A = np.vstack([beds_rest[i][:, 0], np.ones(len(beds_rest[i]))]).T
        result = np.linalg.lstsq(A, beds_rest[i][:, 1], rcond=None)
        m, b = result[0]
        beds_rest[i][:, 1] = m * beds_rest[i][:, 0] + b

    return beds_rest, xti, yti                  

def deform_beds(beds, beds_obs, tpar, sinc):
    """
    Deform the beds using trishear parameters.
    
    Parameters
    ----------
    beds: list of beds to deform.
    beds_obs: list of observed beds.
    tpar: 7-element list with trishear parameters
          Order: xtf, ytf, ramp, ps, tra, slip, c
    sinc: slip increment.

    Returns
    -------
    beds_def: list of deformed beds.
    xtf: x-coordinate of the final fault tip.
    ytf: y-coordinate of the final fault tip.
    """
     # minimum and maximum x and y-coordinates of the beds
    x_min = min([min(bed[:, 0]) for bed in beds_obs])
    x_max = max([max(bed[:, 0]) for bed in beds_obs])
    y_min = min([min(bed[:, 1]) for bed in beds_obs])
    y_max = max([max(bed[:, 1]) for bed in beds_obs])
    # extent of section
    extent_x = x_max - x_min
    extent_y = y_max - y_min   
    
    # model parameters
    xti = tpar[0] # x at fault tip
    yti = tpar[1] # y at fault tip
    ramp = tpar[2] # ramp angle
    ps = tpar[3] # P/S
    tra = tpar[4] # trishear angle
    m = np.tan(tra/2)
    slip = tpar[5] # fault slip
    c = tpar[6] # concentration factor
    ninc = round(slip / sinc) # number of slip increments
    sinc = slip/ninc # slip increment

    # transformation matrix from geographic to fault coordinates
    a11 = np.cos(ramp)
    a12 = np.cos(np.pi / 2 - ramp)
    a21 = np.cos(np.pi / 2 + ramp)
    a22 = a11

    # deformed beds
    beds_def = copy.deepcopy(beds)

    # beds transformed to fault coordinates and with origin at fault tip
    beds_transf = []
    for bed in beds_def:
        fx = (bed[:, 0] - xti) * a11 + (bed[:, 1] - yti) * a12
        fy = (bed[:, 0] - xti) * a21 + (bed[:, 1] - yti) * a22
        beds_transf.append(np.column_stack((fx, fy)))

    # create a figure and axis
    fig, ax = plt.subplots()

    # restore the beds
    for i in range(ninc):
        for j in range(len(beds_transf)):
            fx, fy = beds_transf[j][:, 0], beds_transf[j][:, 1]
            for k in range(len(fx)):
                # solve trishear in a coordinate system
                # attached to the fault tip. Note: first
                # move tip and then deform
                xx = fx[k] - (ps * i * np.abs(sinc))
                yy = fy[k]
                # compute trishear velocity
                vx, vy = vel_trishear(xx, yy, sinc, m, c)
                # update fx and fy
                fx[k] = fx[k] + vx
                fy[k] = fy[k] + vy
            # update the transformed bed
            beds_transf[j][:, 0] = fx
            beds_transf[j][:, 1] = fy
            # update the deformed bed
            beds_def[j][:, 0] = (beds_transf[j][:, 0] 
                                  * a11 + beds_transf[j][:, 1] * a21) + xti
            beds_def[j][:, 1] = (beds_transf[j][:, 0] 
                                  * a12 + beds_transf[j][:, 1] * a22) + yti
        # fault tip
        xtf = xti + (ps * i * np.abs(sinc)) * a11
        ytf = yti + (ps * i * np.abs(sinc)) * a12

        # clear previous plot
        ax.clear()
        
        # axis settings
        ax.set_xlim(x_min-extent_x*1.5, x_max+extent_x*0.1)
        ax.set_ylim(y_min-extent_y*1.5, y_max+extent_y*0.1)
        ax.set_aspect("equal")

        # plot the observed beds in gray
        for bed in beds_obs:
            ax.plot(bed[:, 0], bed[:, 1], ".", color="gray", markersize=1)
        # plot the deformed beds in blue
        for bed in beds_def:
            ax.plot(bed[:, 0], bed[:, 1], "b.", markersize=1)
        # plot the fault
        ax.plot([xti, xtf], [yti, ytf], "r-", linewidth=2)
        ax.plot(xtf, ytf, "ro", markersize=5)

        # clear previous plot
        clear_output(wait=True)
        # redisplay the updated plot
        display(fig)

    # prevents Jupyter from displaying another plot
    plt.close(fig) 

    return beds_def, xtf, ytf      

