#!/bin/bash
#SBATCH -p shared				# what type of node to run on
#SBATCH --nodes=1				# number of nodes
#SBATCH --ntasks-per-node=128	# number of processors / node (<=256)
#SBATCH --mem=2G  				# maximum memory allocation
#SBATCH -t 00:05:00				# time requested
#SBATCH -J steady				# name of job
#SBATCH -A mit183				# account to charge
#SBATCH -o output_file.o		# output file
#SBATCH -e error_file.e 		# error file
#SBATCH --export=ALL 			# 

module purge  					# remove loaded modules
module load cpu 				# will run on cpu node
module load slurm 				# load slurm
module load gcc  				# load gnu compilers
module load openmpi 			# load openmpi

casename='steady'				# .re2 filename
echo $casename > SESSION.NAME 	# first line of SESSION.NAME
echo `pwd`'/' >> SESSION.NAME 	# second line of SESSION.NAME

srun -n 128 ./nek5000 > $casename.log 2>&1  	# run