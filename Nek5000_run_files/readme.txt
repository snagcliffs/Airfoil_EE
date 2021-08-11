How to run simulation:

    1) Generate files for mesh, parameters, hstory points, and inlet velocity via running build_case.py

    2) Compile nek code

        On local computer: 
            Run in terminal: makenek steady

        On xsede:
            Run in terminal:
                module load gcc openmpi
                export CC=mpicc
                export FC=mpif77
                makenek steady

    3) Run 
        
        On local computer: 
            nekbmpi steady $num_processors

        On xsede:
            First edit run_case.sh 
            Run in terminal: sbatch run_case.sh

How to change the simulation:

    Changing the geometry:
    The geometry is given by airfoil.geo.

    Changing run parameters:
    Change values in airfoil.par and SIZE

