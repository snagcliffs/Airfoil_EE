import numpy as np
import subprocess
import os
from scipy.signal import gaussian
from scipy.ndimage import filters
from tqdm import tqdm
import pymech.neksuite as nek

def write_inlet_velocity(T,dt,lam,mu):
    """
    Very simple method for creating smoothed OU process for inlet velocity
    """

    m=int(T/dt)
    time = np.arange(m+1)*dt

    du = np.zeros(m)

    window = int(1.0/dt)
    scale = int(0.125/dt)
    gauss_kern = gaussian(window, scale)
    gauss_kern = gauss_kern/np.sum(gauss_kern)

    # Get OU process
    for j in range(m-1):
        du[j+1] = du[j] - lam*du[j]*dt + mu*np.sqrt(dt)*np.random.randn(1)

    du_smooth = filters.convolve1d(du, gauss_kern)
    du_smooth = du_smooth - du_smooth[0]

    with open("du.dat", "w") as ux_file:
        for j in range(m):
            ux_file.write(str(du_smooth[j]) + '\n')

def write_hist_points(n_hist_top, n_hist_bottom, alpha):

    # Parameters for naca 4412 airfoil
    m = 0.04
    p = 0.4
    t = 0.12
    c = 1
    x_nose = -0.25

    with open("airfoil.his", "w") as file:

        file.write(str(n_hist_top + n_hist_bottom) + '\n')

        for j in range(n_hist_top):

            if n_hist_top > 1: x = j/(n_hist_top-1)
            else: x = 0

            # Airfoil thickness
            yt = 5*t*(0.2969*np.sqrt(x)-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)+0.001

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

            file.write(str(xj) + ' ' + str(yj) + ' 0.0 \n')

        for j in range(n_hist_bottom):

            x = (j+1)/(n_hist_bottom+1)

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
            
            file.write(str(xj) + ' ' + str(yj) + ' 0.0 \n')

def parse_history_points(n_hist_top, n_hist_bottom):
    """
    Takes history file written by nek and converts to individual files;
    pres_hist.dat --- (m * n_hist) matrix
    time_hist.dat --- (m * 1)
    hist_points.dat --- (n_hist * 2)

    Currently loads everything into memory.
    """

    n_hist = n_hist_top + n_hist_bottom

    full_hist = np.genfromtxt("airfoil.his", skip_header=n_hist+1)
    m = int(full_hist.shape[0]/n_hist)
    print(full_hist.shape)

    time_hist = np.array([full_hist[j*n_hist, 0] for j in range(m)])
    pres_hist = np.hstack([np.array([np.round(full_hist[j*n_hist+i, 3],5) for j in range(m)]).reshape(m,1) for i in range(n_hist)])
    vort_hist = np.hstack([np.array([np.round(full_hist[j*n_hist+i, 4],5) for j in range(m)]).reshape(m,1) for i in range(n_hist)])

    np.save('./time_hist', time_hist)
    np.save('./pres_hist', pres_hist)
    np.save('./vort_hist', vort_hist)

    with open("airfoil.his", "r+") as file:
        lines = file.readlines()
        file.seek(0)
        for j in range(n_hist):
            file.write(lines[j+1])
        file.truncate()

    subprocess.run("mv airfoil.his ./hist_points.dat", shell=True)

def write_par_files(T,dt,dT,Re,alpha0,hist_freq):
    """
    Write .par file for each of the two meshes

    userParam01 is used by pitch.usr to determine which mesh is being used.  This helps with writing the mass matrices
    forceCoeffs.dat
    """

    par_lines = ['#',\
                 '# nek parameter file',\
                 '#',\
                 '',\
                 '[GENERAL]',\
                 'stopAt = endTime',\
                 'endTime = '+str(T),\
                 'dt = '+str(dt),\
                 'variableDt = no',\
                 '',\
                 'targetCFL = 3.0',\
                 'timeStepper = BDF2',\
                 'extrapolation = OIFS',\
                 'writeControl = runTime',\
                 'writeInterval = '+str(dT),\
                 'dealiasing = yes',\
                 '',\
                 'filtering = explicit',\
                 'filterWeight = 0.02',\
                 'filterCutoffRatio = 0.65',\
                 '',\
                 'userParam01 = '+str(int(T/dt)),\
                 'userParam02 = '+str(int(hist_freq)),\
                 '',\
                 '[PROBLEMTYPE]',\
                 'equation = incompNS',\
                 'stressFormulation = no',\
                 '',\
                 '[PRESSURE]',\
                 'residualTol = 1e-8',\
                 'residualProj = yes',\
                 '',\
                 '[VELOCITY]',\
                 'residualTol = 1e-8',\
                 'residualProj = no',\
                 'density = 1.0',\
                 'viscosity = '+str(1./Re),\
                 'advection = yes']

    # Write airfoil.par
    with open('airfoil.par', 'w') as file:
        file.seek(0)
        for line in par_lines: file.write(line+'\n')
        file.truncate()

def get_data(base_dir, skip = 100, n_files = 1000):

    files = np.sort([f for f in os.listdir(base_dir) if f[:8]=='airfoil0'])
    U = []
    V = []
    P = []
    Vort = []
    time = []
    
    _,Cx,Cy,_,_,_,mass = load_file(base_dir+files[0])

    for j in tqdm(range(n_files)):
                        
        t,u,v,p,T = load_file(base_dir+files[j+skip], return_xy=False)
        time.append(t)
        U.append(u)
        V.append(v)
        P.append(p)
        Vort.append(T)

    return time, mass, Cx, Cy, np.stack(U).T, np.stack(V).T, np.stack(P).T, np.stack(Vort).T

def load_file(file, return_xy=True, return_T = False):
    """
    Load velocity, pressure, and coorinates field from the file
    """

    field = nek.readnek(file)
    
    t = field.time
    nel = len(field.elem) # Number of spectral elements
    nGLL = field.elem[0].vel.shape[3] # Order of the spectral mesh
    n = nel*nGLL**2
    
    Cx = np.array([field.elem[i].pos[0, 0, j, k]
                   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    Cy = np.array([field.elem[i].pos[1, 0, j, k]
                   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])

    u = np.array([field.elem[i].vel[0, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    v = np.array([field.elem[i].vel[1, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    p = np.array([field.elem[i].pres[0, 0, j, k]
            for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    T = np.array([field.elem[i].temp[0, 0, j, k]
            for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    
    if return_xy: return t,Cx,Cy,u,v,p,T
    else: return t,u,v,p,T

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

    for j in range(n):
        dist[j] = np.min(np.linalg.norm(wing_boundary - GLL_points[:,j].reshape(2,1),axis=0))
    
    return dist

def POD(data,mass,weights=None,max_rank=1000):
    
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
    Psi = Psi[:,::-1][:,:max_rank]
    Sigma = Sigma2[::-1][:max_rank]**0.5
    Phi = stacked_data.dot(Psi).dot(np.diag(Sigma**-1))
    
    return mean_flow,Phi,Sigma,Psi

def parse_outfiles(data_dir, save_dir, skip, n_files, data_name, save_locs=False, get_POD=False):


    print('Loading data from nek5000 restart files')
    t,mass,Cx,Cy,U,V,P,Vort = get_data(data_dir, skip = skip, n_files = n_files)

    if save_locs:
        np.save(save_dir+'/Cx', Cx)
        np.save(save_dir+'/Cy', Cy)

    #np.save(save_dir+'/t_'+data_name, t)
    #np.save(save_dir+'/mass_'+data_name, mass)

    if get_POD:
        
        #print('Computing POD weights')
        #D = get_dist(Cx,Cy)
        #weights = np.exp(-5*D)
        #weights = 1/(1+np.exp(500*(D-0.025)))

        print('Taking proper orthogonal decomposition')
        mean_flow, Phi, Sigma, Psi = POD([U, V], mass, max_rank=100)
        
        #POD_dict = {'Phi':Phi,'Sigma':Sigma,'Psi':Psi,'mean_flow':mean_flow}
        #np.save(save_dir+'/Re_'+data_name+'_POD', POD_dict)

        np.save(save_dir+'/Phi2', Phi)
        np.save(save_dir+'/Sigma2', Sigma)
        np.save(save_dir+'/Psi2', Psi)
        np.save(save_dir+'/mean_flow2', mean_flow)

if __name__ == "__main__":

    parse_outfiles('./Re_10000/outfiles/', \
                   './Re_10000', \
                   700, \
                   2000, \
                   '10000', \
                   False, True)

