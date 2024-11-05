import numpy as np
import matplotlib.pyplot as plt

def general_shear(pts,S1,gamma,kk,ninc,ax,d_p=True):
	"""
	general_shear computes displacement paths, kinematic
	vorticity numbers and progressive finite strain 
	history, for general shear with a pure shear stretch,
	no area change, and a single shear strain
	
	USE: paths,wk,pfs = 
		general_shear(pts,S1,gamma,kk,ninc,ax)
	
	pts = npoints x 2 matrix with X1 and X3 coord. of points
	S1 = Pure shear stretch parallel to shear zone
	gamma = Engineering shear strain
	kk = An integer that indicates whether the maximum 
		finite stretch is parallel (kk = 0), or 
		perpendicular (kk=1) to the shear direction
	ninc = number of strain increments
	ax = an array of two axis handles for the plots
	d_p = plot displacement paths (True) or not (False)
	paths = displacement paths of points
	wk = Kinematic vorticity number
	pfs = progressive finite strain history. column 1 =
		orientation of maximum stretch with respect to 
		X1, column 2 = maximum stretch magnitude
	
	NOTE: Intermediate principal stretch is 1.0 (Plane
		strain). Output orientations are in radians
		
	Python function translated from the Matlab function
	GeneralShear in Allmendinger et al. (2012)
	"""
	# compute minimum principal stretch and incr. stretches
	S1_inc =S1**(1.0/ninc)
	S3 =1.0/S1
	S3_inc =S3**(1.0/ninc)
	
	# incremental engineering shear strain
	gamma_inc = gamma/ninc
	
	# initialize displacement paths
	npts = pts.shape[0] # number of points
	paths = np.zeros((ninc+1,npts,2))
	paths[0,:,:] = pts # initial points of paths
	
	# calculate incremental deformation gradient tensor
	# if max. stretch parallel to shear direction Eq. 8.45
	if kk == 0:
		F=np.zeros((2,2))
		F[0,]=[S1_inc, (gamma_inc*(S1_inc-S3_inc))/
			(2.0*np.log(S1_inc))]
		F[1,]=[0.0, S3_inc]
	# if max. stretch perpendicular to shear direction Eq. 8.46
	elif kk == 1:
		F=np.zeros((2,2))
		F[0,]= [S3_inc, (gamma_inc*(S3_inc-S1_inc))/
							(2.0*np.log(S3_inc))]
		F[1,]= [0.0, S1_inc]
	
	# compute displacement paths
	for i in range(npts): # for all points
		for j in range(ninc+1): # for all strain increments
			for k in range(2):
				for L in range(2):
					paths[j,i,k] = F[k,L]*paths[j-1,i,L] + paths[j,i,k]
		# plot displacement path of point
		xx = paths[:,i,0]
		yy = paths[:,i,1]
		if d_p:
			ax[0].plot(xx,yy,".-",color="gray")
	
	# plot initial and final polygons
	inpol = np.zeros((npts+1,2))
	inpol[0:npts,]=paths[0,0:npts,:]
	inpol[npts,] = inpol[0,]
	ax[0].plot(inpol[:,0],inpol[:,1],"b-")
	finpol = np.zeros((npts+1,2))
	finpol[0:npts,]=paths[ninc,0:npts,:]
	finpol[npts,] = finpol[0,]
	ax[0].plot(finpol[:,0],finpol[:,1],"r-")
	
	# set axes
	ax[0].set_xlabel(r"$\mathbf{X_1}$")
	ax[0].set_ylabel(r"$\mathbf{X_3}$")
	ax[0].grid()
	ax[0].axis("equal")
	
	# determine the eigenvectors of the flow (apophyses)
	# since F is not symmetrical, use function eig
	_,V = np.linalg.eig(F)
	theta2 = np.arctan(V[1,1]/V[0,1])
	wk = np.cos(theta2)
	
	# initalize progressive finite strain history. 
	pfs = np.zeros((ninc+1,2))
	# initial state is unknown

	# calculate progressive finite strain history
	for i in range(1,ninc+1):
		# determine the finite deformation gradient tensor
		finF = np.linalg.matrix_power(F, i)
		# determine Green deformation tensor
		G = np.dot(finF,finF.conj().transpose())
		# stretch magnitude and orientation: Maximum 
		# eigenvalue and their corresponding eigenvectors
		# of Green deformation tensor
		D, V = np.linalg.eigh(G)
		pfs[i,0] = np.arctan(V[1,1]/V[0,1])
		pfs[i,1] = np.sqrt(D[1])
	
	# plot progressive finite strain history
	# but don't include the initial state
	ax[1].plot(pfs[1:,0]*180/np.pi,pfs[1:,1],"k.-")
	ax[1].set_xlabel(r"$\Theta\;(\circ)$")
	ax[1].set_ylabel("Maximum finite stretch")
	ax[1].set_xlim(-90,90)
	ax[1].set_ylim(1,max(pfs[:,1])+0.5)
	ax[1].grid()
	
	return paths, wk, pfs