#!/bin/bash
#SBATCH --job-name=run_arrays
#SBATCH --time=14-00:00:00
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-34
#SBATCH --array=0-2  #buglist
#SBATCH --array=15-20,22-26,32-34  #finish 37507488
#SBATCH --array=2,6,8,12,14,21,27-31  #finish 37256394 # for howto archive
#SBATCH --array=1,2,3,8,12,14,21,27-31  #finish 37256394
#SBATCH --array=18-34  #finish 37117747
##SBATCH --array=2,7,8,13,14,17,19
#SBATCH --array=18,20,22,24,26-27,29-33%5
##SBATCH --array=0-34%5  #finish 32117747
#SBATCH --array=4,6-11,13-17,19,21,25,28%1
#SBATCH --array=23,24%1
##SBATCH --array=0-1%1
#SBATCH --output=./run_batch%A.out

date

#sbatch oei.sh $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_JOB_ID $1 $2 $3 $4 $5
sbatch oei.sh $SLURM_ARRAY_TASK_ID 32117747 $1 $2 $3 $4 $5



echo "this is runoei.sh $1 $2 $3 $4 $5 $6 $7"
date
