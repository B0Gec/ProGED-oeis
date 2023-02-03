#!/bin/bash
#SBATCH --job-name=sim
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --cpus-per-task=1
#SBATCH --array=1-28
#SBATCH --output=./results/slurm/parestim_sim/e1/slurm_output_%A_%a.out

cd ./MLJ23/
singularity exec proged_container.sif python3.7 ./src/MLJ_parestim_sim_fit_hpc.py ${SLURM_ARRAY_TASK_ID}