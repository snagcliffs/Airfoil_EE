import numpy as np
import subprocess
import os
from tqdm import tqdm
import pymech.neksuite as nek

def load_file(file, return_xy=True):
	"""
	Load data from nek output file
	"""

	field = nek.readnek(file)
	
	t = field.time
	nel = len(field.elem) # Number of spectral elements
	nGLL = field.elem[0].vel.shape[3] # Order of the spectral mesh
	n = nel*nGLL**2
	
	if return_xy: 

		Cx = np.array([field.elem[i].pos[0, 0, j, k]
					   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
		Cy = np.array([field.elem[i].pos[1, 0, j, k]
					   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
		mass = np.array([field.elem[i].temp[0, 0, j, k]
			   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])

		return Cx,Cy,mass

	else: 
		u = np.array([field.elem[i].vel[0, 0, j, k]
			   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
		v = np.array([field.elem[i].vel[1, 0, j, k]
			   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
		return t,u,v

def get_wing_boundary(alpha=5, n_points=50):
	
	# Parameters for naca 4412 airfoil
	m = 0.04
	p = 0.4
	t = 0.12
	c = 1
	x_nose = -0.25
	
	X = []
	Y = []
	
	for j in range(n_points):

		x = j/(n_points-1)

		# Airfoil thickness
		yt = 5*t*(0.2969*np.sqrt(x)-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)

		# Center coord height
		if x < p:
			yc = m/p**2*(2*p*(x/c)-(x/c)**2)
			dyc = 2*m/p**2*(p-x/c)
		else:
			yc = m/(1-p)**2*(1-2*p+2*p*(x/c)-(x/c)**2)
			dyc = 2*m/(1-p)**2*(p-x/c)

		theta = np.arctan(dyc)
		xu = x - yt*np.sin(theta) + x_nose
		yu = yc + yt*np.cos(theta)

		xj = np.round(xu*np.cos(-alpha*np.pi/180) + yu*np.sin(alpha*np.pi/180), 5)
		yj = np.round(-xu*np.sin(alpha*np.pi/180) + yu*np.cos(alpha*np.pi/180), 5)

		X.append(xj)
		Y.append(yj)

	for j in range(n_points):

		x = 1-(j+1)/n_points # Now going backwards

		# Airfoil thickness
		yt = 5*t*(0.2969*np.sqrt(x)-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)

		# Center coord height
		if x < p:
			yc = m/p**2*(2*p*(x/c)-(x/c)**2)
			dyc = 2*m/(1-p)**2*(p/c-x/c**2)
		else:
			yc = m/(1-p)**2*(1-2*p+2*p*(x/c)-(x/c)**2)
			dyc = 2*m/(1-p)**2*(p/c-x/c**2)

		theta = np.arctan(dyc)
		xb = x + yt*np.sin(theta) + x_nose
		yb = yc - yt*np.cos(theta)

		xj = np.round(xb*np.cos(-alpha*np.pi/180) + yb*np.sin(alpha*np.pi/180), 5)
		yj = np.round(-xb*np.sin(alpha*np.pi/180) + yb*np.cos(alpha*np.pi/180), 5)

		X.append(xj)
		Y.append(yj)
	
	return X,Y

def get_dist(Cx,Cy,n_points=10000):
	"""
	Returns distance of points with coordinate given by Cx, Cy to airfoil points
	"""

	GLL_points = np.vstack([Cx,Cy])
	n = GLL_points.shape[1]
	dist = np.zeros(n)
	wing_boundary = np.vstack([np.array(wb) for wb in get_wing_boundary(n_points=n_points)])

	for j in tqdm(range(n)):
		dist[j] = np.min(np.linalg.norm(wing_boundary - GLL_points[:,j].reshape(2,1),axis=0))
	
	return dist

def main(data_dir, save_dir, skip=1, n_files=4080):

	print('Saving velocity to .npy files')
	T = []

	for j in tqdm(range(n_files)):

		nek_filename = 'airfoil0.f{0:05d}'.format(j+skip+1)

		# numpy index shifted by 1 so that vel(index) is at time dt*index
		np_filename = 'numpy_files/vel{0:05d}'.format(j+skip)

		t,u,v = load_file(data_dir+nek_filename, return_xy=False)
		T.append(T)

		np.save(save_dir + np_filename, np.vstack([u,v]).T)

	# Distance of each grid point to the airfoil
	print('\nComputing distance to airfoil')
	Cx,Cy,mass = load_file(data_dir+'airfoil0.f00001')
	D = get_dist(Cx,Cy)
	np.save(save_dir+'dist', D)
	np.save(save_dir+'numpy_files/snapshot_times', np.array(T))

if __name__ == "__main__":

	main('./outfiles/', \
		 './')

