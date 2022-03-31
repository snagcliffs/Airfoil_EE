import numpy as np
import subprocess
import argparse
from time import time as timer
import pymech.neksuite as nek
from tqdm import tqdm

import sys
sys.path.append('../../P_to_Xi/core')
from FFNN_generator import FFNN_Generator
from FFNN_net import pressure_encoder

# Uncomment lines to use CPU for NN.  
# We're only evaluating the forward model a few times so NN computation time is negligable.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.random.seed(0)

def write_mesh():

    # Convert .geo file to .msh then .re2 and .ma2
    subprocess.run("gmsh -format msh2 -order 2 -2 airfoil.geo", cwd='./nek5000', stdout=subprocess.DEVNULL, shell=True)
    subprocess.run("gmsh2nek << EOF \n2\nairfoil\n0\n EOF", cwd='./nek5000', stdout=subprocess.DEVNULL, shell=True)
    subprocess.run("genmap << EOF \nairfoil\n0.01\n EOF", cwd='./nek5000', stdout=subprocess.DEVNULL, shell=True)

def write_par(T,dt,Re,start_file):
    """
    Write .par file
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
    
    t_file = field.time
    assert t_file == t

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

    # Write data to field file
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

    nek.writenek('./nek5000/NN_restart0.f00000', field)

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

def build_model_and_gen(datapath, rank):

    ##
    ## Set up data generator
    ##
    S = np.arange(50)*2
    eps = 0.1
    delta = 0.1
    l = 1
    dist_func = lambda d : (1-eps)/(1+np.exp(((d-l)/delta).clip(-np.inf, 35))) + eps
    
    gen = FFNN_Generator(datapath, 
         dist_func,
         S=S, 
         m_hist=70, 
         stride=3, 
         batch_size=1000, 
         ind_min=80)

    ##
    ## Load trained network
    ##
    model = pressure_encoder(restart_file = '../../P_to_Xi/saved_models/FFNN_'+str(rank)+'.npy')

    return model, gen

def main(args):

    n_samp = args.n_samp
    rank = args.rank
    restart_dataset = args.restart_dataset
    tau = args.tau
    dt = args.dt
    Re = args.Re
    transient = args.transient
    datapath = args.datapath
    savepath = args.savepath
    save_file = args.save_file+'_'+restart_dataset+'_r'+str(rank)
    summ_file = args.summ_file+'_'+restart_dataset+'_r'+str(rank)+'.txt'
    test_full_restart = args.test_full_restart

    # Write run parameters to summ_file
    subprocess.run("mkdir "+savepath, shell=True, stdout=subprocess.DEVNULL)
    with open(savepath + summ_file, 'a') as file:
        file.write('Testing accuracy of low dimensional initializations.\nRun arguments:\n')
        for arg in vars(args):
            if getattr(args, arg) is not None: file.write(str(arg)+'\t'+str(getattr(args, arg))+'\n')
        file.write('\n\n')

    # Load NN
    # For now parameters and weight files are written in code directly
    model, gen = build_model_and_gen(datapath, rank)

    # Load force coeffs from original run
    forceCoeffs = np.load(datapath+'forceCoeffs.npy')
    fC_time = forceCoeffs[100:,0]
    Cd = forceCoeffs[100:,1]
    Cl = forceCoeffs[100:,2]

    # Mass matrx for inner products and mean flow
    mass = np.load(datapath+'mass.npy')
    n = mass.size
    mass = np.stack([mass,mass]).reshape(n,2)

    # Note that result summary files did not use order='F' and thus have artificially low reported initial 
    # velocity residual.  This does not affect other results.
    mean_flow = np.load('../../../POD/POD_files/mean_flow.npy').reshape(n,2,order='F')
    
    snapshot_times = gen.snapshot_times

    if restart_dataset == 'train_val':
        # Sample from the mixed training / validation dataset
        valid_inds = list(gen.train_inds) + list(gen.val_inds)
        starting_inds_course = np.random.choice(valid_inds, n_samp, replace = False)
    
    else:
        # Sample from the test dataset
        # valid_inds = [j for j in gen.test_inds if snapshot_times[j] <= np.max(snapshot_times) - tau]
        # starting_inds_course = np.random.choice(valid_inds, n_samp, replace = False)

        # Or take even range from within test set
        # This allows for more consistency comparing to POD
        starting_times_course = np.linspace(870.25,1020-tau-1,n_samp)
        starting_inds_course = [np.argmin(np.abs(tt - snapshot_times)) for tt in starting_times_course]

    starting_inds_fine = [np.argmin(np.abs(fC_time - snapshot_times[j])) for j in starting_inds_course]

    # Dictionary to store all results
    results = {'starting_inds_fine' : starting_inds_fine,
               'starting_inds_course' : starting_inds_course,
               'tau' : tau,
               'dt' : dt,
               'Cd_true' : [],
               'Cl_true' : [], 
               'Cd_NN' : [],
               'Cl_NN' : []}

    if test_full_restart:
        results['Cd_full'] = []
        results['Cl_full'] = []

    # Convert gmsh file into re2
    write_mesh()

    for j in tqdm(range(n_samp)):

        print('Starting simulations from time:'+"{:8.5f}".format(gen.snapshot_times[starting_inds_course[j]]))

        # Get predicted future drag from full restart
        p_hist, _, vel_full = gen.get_snapshot(starting_inds_course[j])
        vel_NN = model.reconstruct(p_hist).numpy()[0,...]
        u_full = vel_full[:,0]
        v_full = vel_full[:,1]
        u_NN = vel_NN[:,0]
        v_NN = vel_NN[:,1]

        if test_full_restart:
            print('Testing full restart from airfoil0.f{0:05d}'.format(starting_inds_course[j]+gen.ind_min+2))
            start_time = timer()
            start_file = '../../../Re_17500/outfiles/'+'airfoil0.f{0:05d}'.format(starting_inds_course[j]+gen.ind_min+2)
            write_par(tau,dt,Re,start_file)
            Cd_full, Cl_full = run_nek([args.tau,args.dt,args.Re,args.ncpu])
            results['Cd_full'].append(Cd_full)
            results['Cl_full'].append(Cl_full)
            full_restart_time = timer()-start_time
            clean_run()

        # Get predicted future drag from NN
        print('Running simulation with NN predicted initial condition.')
        write_IC(u_NN, v_NN)
        start_file = 'NN_restart0.f00000'
        write_par(tau,dt,Re,start_file)

        Cd_NN, Cl_NN = run_nek([args.tau,args.dt,args.Re,args.ncpu])
        results['Cd_NN'].append(Cd_NN)
        results['Cl_NN'].append(Cl_NN)
        clean_run()

        # Record initial NN residual and normalized MSE
        initial_NN_residual = vel_full - vel_NN
        initial_fluctuation_vel = vel_full-mean_flow
        initial_residual_NMSE = np.linalg.norm(np.multiply(mass**0.5,initial_NN_residual))**2 / \
                                np.linalg.norm(np.multiply(mass**0.5,initial_fluctuation_vel))**2

        # Values from original simulation
        Cd_true = Cd[starting_inds_fine[j]:starting_inds_fine[j]+int(tau/dt)+1]
        Cl_true = Cl[starting_inds_fine[j]:starting_inds_fine[j]+int(tau/dt)+1]
        results['Cd_true'].append(Cd_true)
        results['Cl_true'].append(Cl_true)

        # Write summary statistics to summ_file
        with open(savepath + summ_file, 'a') as file:

            file.write('Sample '+str(j+1)+' of '+str(n_samp)+'.\n')
            file.write('#\n')
            file.write('NN initial normalized velocity residual: '+str(np.round(initial_residual_NMSE, 5))+'\n')
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

            Cd_NN = results['Cd_NN'][-1]
            Cl_NN = results['Cl_NN'][-1]
            file.write("NN:   "+"{:8.5f}".format(np.mean(Cd_NN))+"  {:8.5f}".format(np.mean(Cl_NN))
                          +"  {:8.5f}".format(np.max(Cd_NN))+"  {:8.5f}".format(np.max(Cl_NN))
                          +"  {:8.5f}".format(np.min(Cd_NN))+"  {:8.5f}".format(np.min(Cl_NN))
                          +"  {:8.5f}".format(np.std(Cd_NN))+"  {:8.5f}".format(np.std(Cl_NN))+'\n')

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

            Cd_NN = results['Cd_NN'][-1][transient:]
            Cl_NN = results['Cl_NN'][-1][transient:]
            file.write("NN:   "+"{:8.5f}".format(np.mean(Cd_NN))+"  {:8.5f}".format(np.mean(Cl_NN))
                          +"  {:8.5f}".format(np.max(Cd_NN))+"  {:8.5f}".format(np.max(Cl_NN))
                          +"  {:8.5f}".format(np.min(Cd_NN))+"  {:8.5f}".format(np.min(Cl_NN))
                          +"  {:8.5f}".format(np.std(Cd_NN))+"  {:8.5f}".format(np.std(Cl_NN))+'\n')

            file.write('\n\n')

        # Save results to file
        np.save(savepath+save_file,results)

    clean_all()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--restart_dataset', default='test', type=str, help='Restart dataset [test, train_val]')
    parser.add_argument('--rank', default=32, type=int, help='Rank of NN to use [8,16,32,64]')

    # Verify accuracy of full restart?
    parser.add_argument('--test_full_restart', default=False, type=bool, help='Verify restart from field files?')

    # Number of trials
    parser.add_argument('--n_samp', default=5, type=int, help='Number of samples to check')

    # Run parameters
    parser.add_argument('--tau', default=1.0, type=float, help='Length of simulation')
    parser.add_argument('--dt', default=0.001, type=float, help='Simulation timestep')
    parser.add_argument('--Re', default=17500, type=float, help='Reynolds number')
    parser.add_argument('--transient', default=500, type=int, help='Number of initial timesteps to ignore in statistics')
   
    # Location of data
    parser.add_argument('--datapath', default='../../../Re_17500/', type=str, help='Data location')

    # Number of processors
    parser.add_argument('--ncpu', default=16, type=int, help='Number of processors')

    # Results filename
    parser.add_argument('--savepath', default='../Results/', type=str, help='Saved results location')
    parser.add_argument('--save_file', default='results', type=str, help='Saved results name')
    parser.add_argument('--summ_file', default='summary', type=str, help='Print statistics of each run')

    args = parser.parse_args()

    main(args)





