#!/bin/bash
#SBATCH --job-name=sim
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=3GB
#SBATCH --array=1-2
#SBATCH --output=./results/slurm/sysident_num/e1/slurm_output_%A_%a.out

cd ./MLJ23/
singularity exec proged_container.sif python3.7 ./src/sysident_num_fit_hpc.py ${SLURM_ARRAY_TASK_ID}

#3480 jobs 