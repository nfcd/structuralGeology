import numpy as np

def length_strain(l_i, l_f):
	"""
	Calculates elongation (e), stretch (S), lambda (λ),
	and lambda prime (λ') from initial (l_f) and 
	final (l_f) lengths
	"""
	# calculate elongation
	e = (l_f - l_i) / l_i
	# calculate stretch
	S = 1 + e
	# calculate lambda
	lam = S ** 2.0
	# calculate lambda prime
	lam_prime = 1.0 / lam

	return e, S, lam, lam_prime

def length_strain_from_line(line_i, line_f):
	"""
	Calculates elongation (e), stretch (S), lambda (λ),
	and lambda prime (λ') from initial (line_i) and
	final (line_f) line geometry. The lines are defined 
	by two points in 2D or 3D space. The points are 
	defined by a 2 x 2 or 2 x 3 numpy array.
	"""
	# calculate initial length
	l_i = np.linalg.norm(line_i[1] - line_i[0])
	# calculate final length
	l_f = np.linalg.norm(line_f[1] - line_f[0])
	# calculate elongation, stretch, lambda, and lambda prime
	e, S, lam, lam_prime = length_strain(l_i, l_f)

	return e, S, lam, lam_prime
	
def shear_strain(psi):
	"""
	Calculates shear strain (gamma) from 
	angular shear (psi) in radians
	"""
	# return shear strain
	return np.tan(psi)

def shear_strain_from_lines(line_1, line_2):
	"""
	Calculates shear strain (gamma) from two lines
	that were originally orthogonal. The lines are
	defined by two points in 2D space. The points
	are defined by a 2 x 2 numpy array.
	"""
	# calculate the direction of the first line
	dir_1 = line_1[1] - line_1[0]
	# calculate the normal of the first line
	norm_1 = np.array([dir_1[1], -dir_1[0]])
	# calculate the direction of the second line
	dir_2 = line_2[1] - line_2[0]
	# calculate the angle between the normal of the first line
	# and the direction of the second line
	psi = np.arccos(np.dot(norm_1, dir_2) / (np.linalg.norm(norm_1) * np.linalg.norm(dir_2)))
	# calculate shear strain
	gamma = shear_strain(psi)

	return gamma

def area_stretch(a_i, a_f):
	"""
	Calculates area stretch from initial (a_i) and 
	final (a_f) areas
	"""
	# return area stretch
	return a_f / a_i

def area_stretch_from_S(S1, S2):
	"""
	Calculates area stretch from 
	two orthogonal stretches (S)
	"""
	# return area stretch
	return S1 * S2

def volume_stretch(v_i, v_f):
	"""
	Calculates volume stretch from initial (v_i) and 
	final (v_f) volumes
	"""
	# return volume stretch
	return v_f / v_i

def volume_stretch_from_S(S1, S2, S3):
	"""
	Calculates volume stretch from 
	three orthogonal stretches (S)
	"""
	# return volume stretch
	return S1 * S2 * S3