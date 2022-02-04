#!/bin/sh
#SBATCH --partition=smallmem,serial,parallel
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=126000
#SBATCH -t 05-00:00:00
#SBATCH --output=run_dmft.out

srun hostname
MKL_NUM_THREADS=7 OMP_NUM_THREADS=7 mpirun -np 16 python -u run_dmft.py
