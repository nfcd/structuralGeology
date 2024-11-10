import numpy as np

def slip_tendency(p_stress, pole):
    """
    Calculate the slip tendency on a plane given 
    the principal stresses, magnitude and orientation,
    and the pole to the plane.
    Input:
    p_stress = 3 x 4 array with the principal stresses
    This array has the following:
        row 1 = maximum stress and direction cosines (NED)
        row 2 = intermediate stress and direction cosines (NED)
        row 3 = minimum stress magnitude and direction cosines (NED)
    pole = 1 x 3 array with the direction cosines (NED) of the pole to the plane
    Output:
    tau = shear traction
    sigma = normal traction
    Ts = slip tendency
    """
    # magnitude of the principal stresses
    s1 = p_stress[0,0]
    s2 = p_stress[1,0]
    s3 = p_stress[2,0]
    
    # calculate direction cosines of the principal stress axes
    s1_dc = p_stress[0,1:]
    s2_dc = p_stress[1,1:]
    s3_dc = p_stress[2,1:]

    # calculate the direction cosines of the pole to the plane
    # with respect to the principal stress axes
    l = np.dot(s1_dc,pole)
    m = np.dot(s2_dc,pole)
    n = np.dot(s3_dc,pole)

    # calculate the slip tendency
    sigma = s1 * l**2 + s2 * m**2 + s3 * n**2
    tau = np.sqrt((s1-s2)**2 * l**2 * m**2 + 
                      (s2-s3)**2 * m**2 * n**2 + 
                      (s3-s1)**2 * n**2 * l**2)
    
    # slip tendency
    Ts = tau/sigma

    return sigma, tau, Ts