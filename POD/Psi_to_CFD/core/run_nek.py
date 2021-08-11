import numpy as np
import subprocess
import argparse
from time import time as timer
import pymech.neksuite as nek

np.random.seed(0)

def write_mesh():

    # Convert .geo file to .msh then .re2 and .ma2
    subprocess.run("gmsh -format msh2 -order 2 -2 airfoil.geo", cwd='./nek5000', stdout=subprocess.DEVNULL, shell=True)
    subprocess.run("gmsh2nek << EOF \n2\nairfoil\n0\n EOF", cwd='./nek5000', stdout=subprocess.DEVNULL, shell=True)
    subprocess.run("genmap << EOF \nairfoil\n0.01\n EOF", cwd='./nek5000', stdout=subprocess.DEVNULL, shell=True)

def write_par(T,dt,dT,Re,start_file='POD_restart0.f00000'):
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
                 'startFrom = '+start_file,\
                 'stopAt = numSteps',\
                 'numSteps = '+str(int(T/dt)),\
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
    with open('./nek5000/airfoil.par', 'w') as file:
        file.seek(0)
        for line in par_lines: file.write(line+'\n')
        file.truncate()

def write_IC(X,Phi,mean_flow):

    n = int(Phi.shape[0]/2)
    r = Phi.shape[1]

    restart_U = mean_flow[:n,:].reshape(n,1) + Phi[:n,:] @ X.reshape(r,1)
    restart_V = mean_flow[n:2*n,:].reshape(n,1) + Phi[n:2*n,:] @ X.reshape(r,1)

    # Load blank field file with correct geometry
    field = nek.readnek('./nek5000/airfoil_blank0.f00000')

    # Write data from POD expansion to field file
    field.time = 0
    nel = len(field.elem) # Number of spectral elements
    nGLL = field.elem[0].vel.shape[3] # Order of the spectral mesh
    assert(n == nel*nGLL**2)

    for k in range(nGLL):
        for j in range(nGLL):
            for i in range(nel):
                ind = nGLL**2*i + nGLL*j + k
                field.elem[i].vel[0, 0, j, k] = restart_U[ind]
                field.elem[i].vel[1, 0, j, k] = restart_V[ind]

    # Save to file
    nek.writenek('./nek5000/POD_restart0.f00000', field)

def run_sim(ncpu):
    subprocess.run('touch forceCoeffs.dat', cwd='./nek5000', shell=True)
    subprocess.run("makenek steady", cwd='./nek5000', stdout=subprocess.DEVNULL, shell=True)
    run_cmd = "nekmpi airfoil "+str(ncpu)
    with open('logfile.txt', "w") as outfile:
        subprocess.run(run_cmd, 
                       cwd='./nek5000', 
                       shell=True, 
                       executable="/bin/bash", 
                       stdout=outfile,
                       timeout=5400)

def build_and_run(X, Phi, mean_flow, run_params, clean_files = True):

    T, dT, dt, Re, ncpu = run_params

    if dT is None: dT = 2*T # don't save any field files
    start_time = timer()

    write_IC(X, Phi, mean_flow)
    write_mesh()
    write_par(T,dt,dT,Re)
    run_sim(ncpu)

    # Read Cd, Cd from forceCoeffs.dat
    forceCoeffs = np.genfromtxt('./nek5000/forceCoeffs.dat')[:-1,:]
    sim_time = forceCoeffs[:,0]
    Cd = forceCoeffs[:,1]
    Cl = forceCoeffs[:,2]

    # If Cl or Cd contain nans then do not delete logfiles
    # Uncomment this line for testing.  Not during BO though since we need to rewrite forces.
    #if np.isnan(forceCoeffs).any(): clean_files = False

    if clean_files:
        subprocess.run("./clean.sh", cwd='./nek5000', shell=True)
        subprocess.run("rm logfile.txt", shell=True)

    print('Completed simulation.  Time:', str(np.round((timer()-start_time)/60, 3)), 'minutes')
    return sim_time,Cd,Cl

def main(args):
    """
    
    """

    POD_prefix = args.POD_prefix
    POD_dir = '../../POD_files/'
    force_dir = '../../../Re_17500/'

    # Load POD data
    Phi = np.load(POD_dir+POD_prefix+'_Phi.npy')[:,:args.rank]
    Sigma = np.load(POD_dir+POD_prefix+'_Sigma.npy')[:args.rank]
    Psi = np.load(POD_dir+POD_prefix+'_Psi.npy')[:,:args.rank]
    mean_flow = np.load(POD_dir+'mean_flow.npy')
    t_POD = np.load(POD_dir+'t_POD.npy')

    # Load long run forces
    forceCoeffs = np.load(force_dir+'forceCoeffs.npy')
    t_long = forceCoeffs[100:,0]
    Cd_long = forceCoeffs[100:,1]
    Cl_long = forceCoeffs[100:,2]

    # Random IC
    idt = np.random.choice(Psi.shape[0])
    X = Sigma*Psi[idt,:]
    
    # Run simulation
    T = args.T
    dT = args.dT
    dt = args.dt
    Re = args.Re
    ncpu = args.ncpu
    run_params = (T, dT, dt, Re, ncpu)
    sim_time,Cd,Cl = build_and_run(X, Phi, mean_flow, run_params, clean_files=False)

    # Get full rank force data to compare to
    t0 = t_POD[idt]
    ind0 = np.argmin(np.abs(t_long - t0))
    Cd_full = Cd_long[ind0:ind0+int(T/dt)]
    Cl_full = Cl_long[ind0:ind0+int(T/dt)]

    print('Completed test run')
    print('Cd MSE:', np.mean((Cd-Cd_full)**2))
    print('Cl MSE:', np.mean((Cl-Cl_full)**2))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Which POD to use
    parser.add_argument('--POD_prefix', default='midEps', type=str, help='Prefix for POD file')

    # Rank of POD expansion to test
    parser.add_argument('--rank', default=250, type=int, help='Rank of POD expansion')

    # Run and step lengths
    parser.add_argument('--T', default=10, type=float, help='Length of simulation')
    parser.add_argument('--dt', default=0.001, type=float, help='Simulation timestep')
    parser.add_argument('--dT', default=20, type=float, help='IO timestep')

    # Reynolds
    parser.add_argument('--Re', default=17500, type=float, help='How many Re to run')
    
    # Number of processors
    parser.add_argument('--ncpu', default=16, type=int, help='Number of proceesors')

    args = parser.parse_args()

    main(args)
