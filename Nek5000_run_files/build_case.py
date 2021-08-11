import numpy as np
import subprocess
import os
import argparse
from IO import *

def main(args):
    """
    
    """

    T = args.T
    dt = args.dt
    dT = args.dT
    alpha0 = args.alpha0
    Re = args.Re

    hist_freq = args.hist_freq
    n_hist_top = args.n_hist_top
    n_hist_bottom = args.n_hist_bottom
    
    lam = args.lam
    mu = args.mu

    # Convert .geo files to .msh, re2, and ma2
    subprocess.run("gmsh -format msh2 -order 2 -2 airfoil.geo", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("gmsh2nek << EOF \n2\nairfoil\n0\n EOF", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("genmap << EOF \nairfoil\n0.01\n EOF", shell=True, stdout=subprocess.DEVNULL)

    # Write and adjust input files as needed
    write_par_files(T,dt,dT,Re,alpha0,hist_freq)
    write_inlet_velocity(T,dt,lam,mu)
    write_hist_points(n_hist_top, n_hist_bottom, alpha0)

    # Make files to dump cylinder data and mass
    subprocess.run('touch forceCoeffs.dat', shell=True)

    # Build and run
    # These commands were run on xsede not using python code
    # subprocess.run("module load gcc openmpi", shell=True, stdout=subprocess.DEVNULL)
    # subprocess.run("export CC=mpicc", shell=True, stdout=subprocess.DEVNULL)
    # subprocess.run("export FC=mpif77", shell=True, stdout=subprocess.DEVNULL)
    # subprocess.run("makenek steady", shell=True, stdout=subprocess.DEVNULL)

    """
    Add in any post-processing here
    """

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Run and step lengths
    parser.add_argument('--T', default=10, type=float, help='Length of simulation')
    parser.add_argument('--dt', default=0.001, type=float, help='Simulation timestep')
    parser.add_argument('--dT', default=0.25, type=float, help='IO timestep')
    parser.add_argument('--alpha0', default=5, type=float, help='Initial angle of attack')

    # Number of history points to track
    parser.add_argument('--hist_freq', default=int(10), type=int, help='Number of steps between writing history points')
    parser.add_argument('--n_hist_top', default=51, type=int, help='Number of history points along top of airfoil')
    parser.add_argument('--n_hist_bottom', default=49, type=int, help='Number of history points along bottom of airfoil')

    # Reynolds
    parser.add_argument('--Re', default=17500, type=int, help='Reynolds number')
    
    # Parameters for stochastic inlet velocity
    parser.add_argument('--lam', default=0., type=float, help='Restorative force for OU inlet velocity')
    parser.add_argument('--mu', default=0., type=float, help='Noise variance for OU inlet velocity')

    args = parser.parse_args()

    main(args)
