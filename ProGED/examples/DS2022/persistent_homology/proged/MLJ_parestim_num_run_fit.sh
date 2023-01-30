#!/bin/bash
#SBATCH --job-name=nume1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=3GB
#SBATCH --array=1-312
#SBATCH --output=./MLJ23/results/slurm/parestim_num/e1/slurm_output_%A_%a.out

cd ./MLJ23/

JOBINDEX=$((SLURM_ARRAY_TASK_ID))

singularity exec proged_container_310.sif python3.10 ./src/MLJ_parestim_num_fit_hpc.py ${JOBINDEX}