#!/bin/bash
#SBATCH --job-name=numdiff
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --cpus-per-task=1
#SBATCH --array=1-2
#SBATCH --output=./numdiff/identification_slurm/v1/slurm_output_%A_%a.out

cd ./MLJ23/proged/
singularity exec ./code/proged_container.sif python3.7 ./code/MLJ_test_structures.py ${SLURM_ARRAY_TASK_ID}