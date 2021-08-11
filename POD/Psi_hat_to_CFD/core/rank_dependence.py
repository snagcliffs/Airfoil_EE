import numpy as np
import subprocess
import argparse
from time import time as timer
import pymech.neksuite as nek
from tqdm import tqdm

np.random.seed(0)

def write_mesh():

    # Convert .geo file to .msh then .re2 and .ma2
    subprocess.run("gmsh -format msh2 -order 2 -2 airfoil.geo", cwd='./nek5000', stdout=subprocess.DEVNULL, shell=True)
    subprocess.run("gmsh2nek << EOF \n2\nairfoil\n0\n EOF", cwd='./nek5000', stdout=subprocess.DEVNULL, shell=True)
    subprocess.run("genmap << EOF \nairfoil\n0.01\n EOF", cwd='./nek5000', stdout=subprocess.DEVNULL, shell=True)

def write_par(T,dt,Re,start_file):
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
                 'writeInterval = 1000',\
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

def load_uv(datapath, t):

    file = datapath+'outfiles/'+'airfoil0.f{0:05d}'.format(int(t*4)+1)
    field = nek.readnek(file)
    
    nel = len(field.elem) # Number of spectral elements
    nGLL = field.elem[0].vel.shape[3] # Order of the spectral mesh
    n = nel*nGLL**2

    u = np.array([field.elem[i].vel[0, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    v = np.array([field.elem[i].vel[1, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    
    return u,v

def write_IC(u, v, p=None):

    n = u.size

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
                field.elem[i].vel[0, 0, j, k] = u[ind]
                field.elem[i].vel[1, 0, j, k] = v[ind]
                if p is not None: field.elem[i].pres[0, 0, j, k] = p[ind]

    nek.writenek('./nek5000/POD_restart0.f00000', field)

def run_nek(run_params):

    T, dt, Re, ncpu = run_params

    start_time = timer()

    # Make nek executable
    subprocess.run("makenek steady", 
                   cwd='./nek5000', 
                   stdout=subprocess.DEVNULL, 
                   shell=True)

    # Run.  Logging in ./logfile.txt
    run_cmd = "nekmpi airfoil "+str(ncpu)
    with open('logfile.txt', "w") as outfile:
        subprocess.run(run_cmd, 
                       cwd='./nek5000', 
                       shell=True, 
                       executable="/bin/bash", 
                       stdout=outfile,
                       timeout=1800)

    # Read Cd, Cl from forceCoeffs.dat
    forceCoeffs = np.genfromtxt('./nek5000/forceCoeffs.dat')
    Cd = forceCoeffs[:,1]
    Cl = forceCoeffs[:,2]

    # If Cl or Cd contain nans then do not delete logfiles
    # Uncomment this line for testing.  Not during full runs though since we need to rewrite forces.
    # if np.isnan(forceCoeffs).any(): clean_files = False

    print('Completed simulation.  Time:', str(np.round((timer()-start_time)/60, 3)), 'minutes')

    return Cd, Cl

def clean_run():
    """
    Remove nek output files but leave geometry
    """
    subprocess.run("rm *airfoil0.f*", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm *.dat", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm -rf obj", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm nek5000", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm makefile", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm *logfile.txt", shell=True, stdout=subprocess.DEVNULL)

def clean_all(): 
    """
    Clean files at end of simulation
    """
    subprocess.run("rm *airfoil0.f*", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL) 
    subprocess.run("rm *.dat", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL) 
    subprocess.run("rm *.msh", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL) 
    subprocess.run("rm *.par", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL) 
    subprocess.run("rm *.ma2", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL) 
    subprocess.run("rm *.re2", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL) 
    subprocess.run("rm *.f", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL) 
    subprocess.run("rm *log", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm SESSION.NAME", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm -rf obj", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm nek5000", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm makefile", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm -rf __pycache__", cwd='./nek5000', shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("rm *logfile.txt", shell=True, stdout=subprocess.DEVNULL)

def main(args):

    ranks = args.ranks
    n_samp = args.n_samp
    tau = args.tau
    dt = args.dt
    Re = args.Re
    transient = args.transient
    datapath = args.datapath
    POD_path = args.POD_path
    POD_est_path = args.POD_est_path
    POD_prefix = args.POD_prefix
    savepath = args.savepath
    save_file = args.save_file+'_'+POD_prefix
    summ_file = args.summ_file+'_'+POD_prefix+'.txt'
    test_full_restart = args.test_full_restart

    subprocess.run("mkdir "+savepath, shell=True, stdout=subprocess.DEVNULL)

    # Write run parameters to summ_file
    with open(savepath + summ_file, 'a') as file:

        file.write('Testing accuracy of low dimensional initializations.\nRun arguments:\n')
        for arg in vars(args):
            if getattr(args, arg) is not None: file.write(str(arg)+'\t'+str(getattr(args, arg))+'\n')
        file.write('\n\n')

    # Load POD 
    POD_time = np.load(POD_est_path+'Psi_32.npy',allow_pickle=True).item()['time']
    Phi = np.load(POD_path+POD_prefix+'_Phi.npy')[:,:np.max(ranks)]
    Psi_true = np.load(POD_path+POD_prefix+'_Psi.npy')[:,:np.max(ranks)]
    Sigma = np.load(POD_path+POD_prefix+'_Sigma.npy')[:np.max(ranks)]
    Psis = {}
    for r in ranks:
        Psis[r] = np.load(POD_est_path+'Psi_'+str(r)+'.npy',allow_pickle=True).item()['Psi']
        Psis[r] = Psis[r] * np.std(Psi_true, axis=0)[:r]
    mean_flow = np.load(POD_path+'mean_flow.npy')
    n = int(Phi.shape[0]/2)

    # Load force coeffs from original run
    forceCoeffs = np.load(datapath+'forceCoeffs.npy')
    fC_time = forceCoeffs[100:,0]
    Cd = forceCoeffs[100:,1]
    Cl = forceCoeffs[100:,2]

    # Mass matrx for inner products
    mass = np.load(datapath+'mass.npy')
    mass = np.concatenate([mass,mass])

    # Sample starting points randomly (not currently used)
    # max_POD_sample = np.max(np.where(POD_time+tau < np.max(fC_time)))
    # starting_inds_POD = np.random.choice(len(POD_time[:max_POD_sample]), n_samp, replace=False)

    # Sample uniformly to allow for easier comparrison
    starting_times_POD = np.round(np.linspace(870.25,1020-tau-1,n_samp)*4)/4
    starting_inds_POD = [np.argmin(np.abs(tt - POD_time)) for tt in starting_times_POD]
    starting_inds_fine = [np.argmin(np.abs(POD_time[j] - fC_time)) for j in starting_inds_POD]

    # Dictionary to store all results
    results = {'starting_inds_fine' : starting_inds_fine,
               'starting_inds_POD' : starting_inds_POD,
               'tau' : tau,
               'dt' : dt,
               'Cd_true' : [],
               'Cl_true' : []}

    if test_full_restart:
        results['Cd_full'] = []
        results['Cl_full'] = []

    for r in ranks:
       results['Cd_'+str(r)] = []
       results['Cl_'+str(r)] = []

    # Convert gmsh file into re2
    write_mesh()

    for j in tqdm(range(n_samp)):

        print('Starting simulations from time:'+"{:8.5f}".format(POD_time[starting_inds_POD[j]]))

        # Get predicted future drag from full restart
        u_full, v_full = load_uv(datapath, POD_time[starting_inds_POD[j]])

        if test_full_restart:
            print('Testing full restart from airfoil0.f{0:05d}'.format(int(POD_time[starting_inds_POD[j]]*4)+1))
            start_time = timer()
            start_file = file = datapath+'outfiles/'+'airfoil0.f{0:05d}'.format(int(POD_time[starting_inds_POD[j]]*4)+1)
            write_par(tau,dt,Re,start_file)
            Cd_full, Cl_full = run_nek([args.tau,args.dt,args.Re,args.ncpu])
            results['Cd_full'].append(Cd_full)
            results['Cl_full'].append(Cl_full)
            full_restart_time = timer()-start_time
            clean_run()

        # Get predicted future drag from POD
        initial_residual_NMSE = []

        for r in ranks:

            print('Running simulation with rank '+str(r)+' initial condition.')
            start_time = timer()
            uv_r = mean_flow + np.multiply(Phi[:,:r], Sigma[:r]) @ Psis[r][starting_inds_POD[j],:].reshape(r,1)
            write_IC(uv_r[:n,0],uv_r[n:,0])
            start_file = 'POD_restart0.f00000'
            write_par(tau,dt,Re,start_file)

            Cd_r, Cl_r = run_nek([args.tau,args.dt,args.Re,args.ncpu])
            results['Cd_'+str(r)].append(Cd_r)
            results['Cl_'+str(r)].append(Cl_r)
            POD_restart_time = timer()-start_time 
            clean_run()

            # Record initial POD residual and normalized MSE
            initial_POD_residual = np.concatenate([u_full,v_full]) - uv_r.flatten()
            initial_fluctuation_vel = np.concatenate([u_full,v_full])-mean_flow.flatten()
            initial_residual_NMSE.append(np.linalg.norm(np.multiply(initial_POD_residual,mass**0.5))**2 / \
                                         np.linalg.norm(np.multiply(initial_fluctuation_vel,mass**0.5))**2)

        # Values from original simulation
        Cd_true = Cd[starting_inds_fine[j]:starting_inds_fine[j]+int(tau/dt)+1]
        Cl_true = Cl[starting_inds_fine[j]:starting_inds_fine[j]+int(tau/dt)+1]
        results['Cd_true'].append(Cd_true)
        results['Cl_true'].append(Cl_true)

        # Write summary statistics to summ_file
        with open(savepath + summ_file, 'a') as file:

            file.write('Sample '+str(j+1)+' of '+str(n_samp)+'.\n')
            file.write('#\n')
            file.write('POD initial normalized velocity residual: \n')
            for j in range(len(ranks)):
                file.write('\trank '+str(ranks[j])+': '+str(np.round(initial_residual_NMSE[j], 5))+'\n')
            file.write('#\n')
            file.write('Statistics of aerodynamic coefficients\n')
            file.write('       mean_Cd   mean_Cl   max_Cd    max_Cl    min_Cd    min_Cl    std_Cd    std_Cl   \n')

            file.write('True: '+"{:8.5f}".format(np.mean(Cd_true))+"  {:8.5f}".format(np.mean(Cl_true))
                          +"  {:8.5f}".format(np.max(Cd_true))+"  {:8.5f}".format(np.max(Cl_true))
                          +"  {:8.5f}".format(np.min(Cd_true))+"  {:8.5f}".format(np.min(Cl_true))
                          +"  {:8.5f}".format(np.std(Cd_true))+"  {:8.5f}".format(np.std(Cl_true))+'\n')

            if test_full_restart:
                file.write('Full: '+"{:8.5f}".format(np.mean(Cd_full))+"  {:8.5f}".format(np.mean(Cl_full))
                              +"  {:8.5f}".format(np.max(Cd_full))+"  {:8.5f}".format(np.max(Cl_full))
                              +"  {:8.5f}".format(np.min(Cd_full))+"  {:8.5f}".format(np.min(Cl_full))
                              +"  {:8.5f}".format(np.std(Cd_full))+"  {:8.5f}".format(np.std(Cl_full))+'\n')
            for r in ranks:
                Cd_r = results['Cd_'+str(r)][-1]
                Cl_r = results['Cl_'+str(r)][-1]
                r_string = str(r)+':'; fill = ' '; align='<'; width=4
                description_string = 'r='+f'{r_string:{fill}{align}{width}}'
                file.write(description_string+"{:8.5f}".format(np.mean(Cd_r))+"  {:8.5f}".format(np.mean(Cl_r))
                              +"  {:8.5f}".format(np.max(Cd_r))+"  {:8.5f}".format(np.max(Cl_r))
                              +"  {:8.5f}".format(np.min(Cd_r))+"  {:8.5f}".format(np.min(Cl_r))
                              +"  {:8.5f}".format(np.std(Cd_r))+"  {:8.5f}".format(np.std(Cl_r))+'\n')

            file.write('#\n')
            file.write('With '+str(transient)+' step / t='+"{:5.3f}".format(dt*transient)+' transient cutoff:\n')
            file.write('       mean_Cd   mean_Cl   max_Cd    max_Cl    min_Cd    min_Cl    std_Cd    std_Cl   \n')

            file.write('True: '+"{:8.5f}".format(np.mean(Cd_true[transient:]))+"  {:8.5f}".format(np.mean(Cl_true[transient:]))
                          +"  {:8.5f}".format(np.max(Cd_true[transient:]))+"  {:8.5f}".format(np.max(Cl_true[transient:]))
                          +"  {:8.5f}".format(np.min(Cd_true[transient:]))+"  {:8.5f}".format(np.min(Cl_true[transient:]))
                          +"  {:8.5f}".format(np.std(Cd_true[transient:]))+"  {:8.5f}".format(np.std(Cl_true[transient:]))+'\n')
            
            if test_full_restart:
                file.write('Full: '+"{:8.5f}".format(np.mean(Cd_full[transient:]))+"  {:8.5f}".format(np.mean(Cl_full[transient:]))
                              +"  {:8.5f}".format(np.max(Cd_full[transient:]))+"  {:8.5f}".format(np.max(Cl_full[transient:]))
                              +"  {:8.5f}".format(np.min(Cd_full[transient:]))+"  {:8.5f}".format(np.min(Cl_full[transient:]))
                              +"  {:8.5f}".format(np.std(Cd_full[transient:]))+"  {:8.5f}".format(np.std(Cl_full[transient:]))+'\n')

            for r in ranks:
                Cd_r = results['Cd_'+str(r)][-1][transient:]
                Cl_r = results['Cl_'+str(r)][-1][transient:]
                r_string = str(r)+':'; fill = ' '; align='<'; width=4
                description_string = 'r='+f'{r_string:{fill}{align}{width}}'
                file.write(description_string+"{:8.5f}".format(np.mean(Cd_r))+"  {:8.5f}".format(np.mean(Cl_r))
                              +"  {:8.5f}".format(np.max(Cd_r))+"  {:8.5f}".format(np.max(Cl_r))
                              +"  {:8.5f}".format(np.min(Cd_r))+"  {:8.5f}".format(np.min(Cl_r))
                              +"  {:8.5f}".format(np.std(Cd_r))+"  {:8.5f}".format(np.std(Cl_r))+'\n')

            file.write('\n\n')

        # Save results to file
        np.save(savepath+save_file,results)

    clean_all()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # POD rank
    parser.add_argument('--ranks', type=float, nargs='+', default=[8,16,32,64])

    # Verify accuracy of full restart?
    parser.add_argument('--test_full_restart', default=False, type=bool, help='Verify restart from field files?')

    # Number of trials
    parser.add_argument('--n_samp', default=50, type=int, help='Number of samples to check')

    # Run parameters
    parser.add_argument('--tau', default=7, type=float, help='Length of simulation')
    parser.add_argument('--dt', default=0.001, type=float, help='Simulation timestep')
    parser.add_argument('--Re', default=17500, type=float, help='Reynolds number')
    parser.add_argument('--transient', default=500, type=int, help='Number of initial timesteps to ignore in statistics')
   
    # Location of POD data
    parser.add_argument('--datapath', default='../../../Re_17500/', type=str, help='Data location')
    parser.add_argument('--POD_path', default='../../POD_files/', type=str, help='Where is POD saved?')
    parser.add_argument('--POD_est_path', default='../../P_to_Psi/dense_Psi_predictions/', type=str, help='Where is POD saved?')
    parser.add_argument('--POD_prefix', default='midEps', type=str, help='full, bdry, close, mid, midEps')

    # Number of processors
    parser.add_argument('--ncpu', default=16, type=int, help='Number of processors')

    # Results filename
    parser.add_argument('--savepath', default='../Results/', type=str, help='Saved results location')
    parser.add_argument('--save_file', default='results', type=str, help='Saved results name')
    parser.add_argument('--summ_file', default='summary', type=str, help='Print statistics of each run')

    args = parser.parse_args()

    main(args)





