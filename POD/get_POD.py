import numpy as np
import subprocess
import os
from tqdm import tqdm
import pymech.neksuite as nek

def get_data(base_dir, skip = 80, n_files = 4001):

    files = np.sort([f for f in os.listdir(base_dir) if f[:8]=='airfoil0'])
    U = []
    V = []
    time = []
    
    Cx,Cy,mass = load_file(base_dir+files[0])

    for j in tqdm(range(n_files)):
                        
        t,u,v = load_file(base_dir+files[j+skip], return_xy=False)
        time.append(t)
        U.append(u)
        V.append(v)

    return time, mass, Cx, Cy, np.stack(U).T, np.stack(V).T

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

def POD(data,mass,weights=None,max_rank=100):
    
    stacked_data = np.vstack(data)
    n,m = stacked_data.shape

    mass = np.concatenate([mass for _ in range(len(data))])
    
    if weights is None: 
        weights = np.ones(n)  
    else:
        weights = np.concatenate([weights for _ in range(len(data))])
        
    mean_flow = stacked_data.mean(axis=1).reshape(n,1)
    stacked_data = stacked_data-mean_flow
    
    C = np.multiply(stacked_data.T,mass*weights).dot(stacked_data)
    Sigma2,Psi = np.linalg.eigh(C)

    # Remove singular component
    Sigma2 = Sigma2.clip(0,np.inf) # numerical error can make some values <0
    rank = np.count_nonzero(Sigma2) # could also treat <epsilon as zero
    Sigma = (Sigma2[::-1][:rank])**0.5
    Psi = Psi[:,::-1][:,:rank]

    # Compute Phi
    Phi = stacked_data.dot(Psi).dot(np.diag(Sigma**-1))
    
    return mean_flow,Phi[:,:max_rank],Sigma,Psi[:,:max_rank]

def parse_outfiles(data_dir, save_dir, max_rank=100, skip=80, n_files=4001):

    print('Loading data from nek5000 restart files')
    t,mass,Cx,Cy,U,V = get_data(data_dir, skip = skip, n_files = n_files)

    print('\nCompleted loading data from', len(t), 'files.')
    print('Min(time)='+str(t[0]))
    print('Max(time)='+str(t[-1]))

    np.save('../Re_17500/Cx', Cx)
    np.save('../Re_17500/Cy', Cy)
    np.save('../Re_17500/mass', mass)
    np.save(save_dir+'t_POD', t)

    # Distance of each grid point to the airfoil
    print('\nComputing distance to airfoil')
    D = get_dist(Cx,Cy)

    #####################################################################################
    #
    # Unweighted POD
    #
    #####################################################################################

    print('\nComputing unweighted POD')
    mean_flow,Phi,Sigma,Psi = POD([U, V], mass, max_rank=max_rank)
    np.save(save_dir+'full_Phi', Phi)
    np.save(save_dir+'full_Sigma', Sigma)
    np.save(save_dir+'full_Psi', Psi)
    np.save(save_dir+'mean_flow', mean_flow)

    #####################################################################################
    #
    # Boundary layer focused POD
    #
    #####################################################################################

    print('Computing boundary layer focused POD')

    width = 0.025
    transition_length = 0.002
    eps = 0.0

    # weight = eps + (1-eps)*exp((dist - width) / transition_length)
    # Clip exponential to avoid overflow
    weights = (1-eps)/(1+np.exp(((D-width)/transition_length).clip(-np.inf, 35)))+eps

    _,Phi,Sigma,Psi = POD([U, V], mass, weights, max_rank=max_rank)
    np.save(save_dir+'bdry_Phi', Phi)
    np.save(save_dir+'bdry_Sigma', Sigma)
    np.save(save_dir+'bdry_Psi', Psi)

    #####################################################################################
    #
    # Close field focused POD
    #
    #####################################################################################

    print('Computing near field focused POD')

    width = 0.1
    transition_length = 0.01
    eps = 0.0

    # weight = eps + (1-eps)*exp((dist - width) / transition_length)
    # Clip exponential to avoid overflow
    weights = (1-eps)/(1+np.exp(((D-width)/transition_length).clip(-np.inf, 35)))+eps

    _,Phi,Sigma,Psi = POD([U, V], mass, weights, max_rank=max_rank)
    np.save(save_dir+'/close_Phi', Phi)
    np.save(save_dir+'/close_Sigma', Sigma)
    np.save(save_dir+'/close_Psi', Psi)

    #####################################################################################
    #
    # Mid field focused POD
    #
    #####################################################################################

    print('Computing mid field focused POD')

    width = 1
    transition_length = 0.1
    eps = 0.0

    # weight = eps + (1-eps)*exp((dist - width) / transition_length)
    # Clip exponential to avoid overflow
    weights = (1-eps)/(1+np.exp(((D-width)/transition_length).clip(-np.inf, 35)))+eps

    _,Phi,Sigma,Psi = POD([U, V], mass, weights, max_rank=max_rank)
    np.save(save_dir+'/mid_Phi', Phi)
    np.save(save_dir+'/mid_Sigma', Sigma)
    np.save(save_dir+'/mid_Psi', Psi)

    #####################################################################################
    #
    # Mid field focused POD with non-zero far field weights
    #
    #####################################################################################

    print('Computing mid field focused POD with decay to non-zero value')

    width = 1
    transition_length = 0.1
    eps = 0.1

    # weight = eps + (1-eps)*exp((dist - width) / transition_length)
    # Clip exponential to avoid overflow
    weights = (1-eps)/(1+np.exp(((D-width)/transition_length).clip(-np.inf, 35)))+eps

    _,Phi,Sigma,Psi = POD([U, V], mass, weights, max_rank=max_rank)
    np.save(save_dir+'/midEps_Phi', Phi)
    np.save(save_dir+'/midEps_Sigma', Sigma)
    np.save(save_dir+'/midEps_Psi', Psi)

if __name__ == "__main__":

    parse_outfiles('../Re_17500/outfiles/', \
                   './POD_files/',
                   500)

