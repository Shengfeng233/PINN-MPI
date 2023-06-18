#!/bin/bash
#SBATCH --job-name=xsf_mpi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --partition=wzhcnormal
#SBATCH -t 36000
#SBATCH --exclusive
export PATH="~/soft/miniconda/bin:$PATH"
source activate pytorch
python pre_train.py
mpiexec -n 40 python main.py