#!/bin/bash
#SBATCH --job-name=obs_1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=3GB
#SBATCH --array=1-2
#SBATCH --output=./MLJ23/results/slurm/parobs_sim/e1/slurm_output_%A_%a.out

JOBINDEX=$((SLURM_ARRAY_TASK_ID))

cd ./MLJ23/
singularity exec proged_container_310.sif python3.10 ./src/parobs_sim_fit_hpc.py ${JOBINDEX}

