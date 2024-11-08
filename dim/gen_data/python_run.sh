#!/bin/bash                                                                 
#SBATCH --job-name=python_cpu   # Job name
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=t.desilva@ufl.edu
#SBATCH --ntasks=1                # Number of MPI ranks
#SBATCH --cpus-per-task=1           # Number of cores per MPI rank
##SBATCH --gpus-per-task=1
#SBATCH --partition=hpg-milan
##SBATCH --nodes=1                  # Number of nodes
#SBATCH --mem-per-cpu=60gb          # Memory per processor
#SBATCH --time=10-00:00:00              # Time limit hrs:min:sec
#SBATCH --output=python_cpu.log     # Standard output and error log
pwd; hostname; date

module purge

export PATH=~/.conda/envs/myEnv/bin:$PATH

python gen_data_1.py
