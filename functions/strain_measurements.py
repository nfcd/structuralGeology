import math

def length_strain(l_i, l_f):
    '''
    Calculates elongation (e), stretch (S), lambda (λ),
    and lambda prime (λ') from initial (l_f) and 
    final (l_f) lengths
    '''
    # calculate elongation
    e = (l_f - l_i) / l_i
    # calculate stretch
    S = 1 + e
    # calculate lambda
    lam = S ** 2.0
    # calculate lambda prime
    lam_prime = 1.0 / lam

    return e, S, lam, lam_prime

def shear_strain(psi):
    '''
    Calculates shear strain (gamma) from 
    angular shear (psi) in radians
    '''
    # return shear strain
    return math.tan(psi)

def area_stretch(a_i, a_f):
    '''
    Calculates area stretch from initial (a_i) and 
    final (a_f) areas
    '''
    # return area stretch
    return a_f / a_i

def area_stretch_from_S(S=[1,1]):
    '''
    Calculates area stretch from 
    two orthogonal stretches (S)
    '''
    # return area stretch
    return S[0] * S[1]

def volume_stretch(v_i, v_f):
    '''
    Calculates volume stretch from initial (v_i) and 
    final (v_f) volumes
    '''
    # return volume stretch
    return v_f / v_i

def volume_stretch_from_S(S=[1,1,1]):
    '''
    Calculates volume stretch from 
    three orthogonal stretches (S)
    '''
    # return volume stretch
    return S[0] * S[1] * S[2]