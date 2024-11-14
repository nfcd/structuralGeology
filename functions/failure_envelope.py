import numpy as np

def sigma_1_optimal(sigma_3, phi, c):
	"""
	Calculates the optimal sigma 1 value
	for failure, given sigma 3, friction angle (phi) 
	in radians, and cohesion (c)
	"""
	# calculate optimal sigma 1
	# Jaeger et al. (2007) Eq. 4.10
	cos_phi = np.cos(phi)
	sin_phi = np.sin(phi)
	sigma_1 = (2*c*(cos_phi/(1-sin_phi)) 
			   + sigma_3*((1+sin_phi)/(1-sin_phi)))
	
	return sigma_1

def sigma_3_optimal(sigma_1, phi, c):
	"""
	Calculates the optimal sigma 3 value
	for failure, given sigma 1, friction angle (phi) 
	in radians, and cohesion (c)
	"""
	# calculate optimal sigma 3
	# Jaeger et al. (2007) Eq. 4.10
	cos_phi = np.cos(phi)
	sin_phi = np.sin(phi)
	sigma_3 = ((sigma_1-2*c*(cos_phi/(1-sin_phi))) 
			   / ((1+sin_phi)/(1-sin_phi)))
	
	return sigma_3

def stress_ratio_for_beta(beta, phi):
	"""
	Calculates the stress ratio (sigma 1 / sigma 3) 
	for a given value of beta and static friction 
	angle (phi) in radians. beta is the angle that
	the plane makes with sigma 1, and the failure
	envelope is for a pre-existing fault (c = 0)
	"""
	# coefficient of static friction
	mu = np.tan(phi)
	# calculate stress ratio
	# Ragan (2009) Eq. 10.11
	R = (1+mu/np.tan(beta)) / (1-mu*np.tan(beta))
	
	return R

def sigma_1_for_beta(sigma_3, beta, phi):
	"""
	Calculates the value of sigma 1 for a given value
	of sigma 3, beta, and static friction angle (phi) 
	in radians. beta is the angle that the plane makes 
	with sigma 1, and the failure envelope is for a 
	pre-existing fault (c = 0)
	"""
	# stress ratio
	R = stress_ratio_for_beta(beta, phi)
	# calculate sigma 1
	sigma_1 = sigma_3 * R

	return sigma_1

def sigma_3_for_beta(sigma_1, beta, phi):
	"""
	Calculates the value of sigma 3 for a given value
	of sigma 1, beta, and static friction angle (phi) 
	in radians. beta is the angle that the plane makes 
	with sigma 1, and the failure envelope is for a 
	pre-existing fault (c = 0)
	"""
	# stress ratio
	R = stress_ratio_for_beta(beta, phi)
	# calculate sigma 3
	sigma_3 = sigma_1 / R

	return sigma_3