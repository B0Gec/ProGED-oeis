#!/bin/bash
#SBATCH --job-name=transfOEIS
##SBATCH --partition=long
##SBATCH --time=14-00:00:00
##SBATCH --time=00:10:00
##SBATCH --time=2-00:00:00
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=5GB
##SBATCH --mem-per-cpu=100MB  ### mavi 100MB
##SBATCH --cpus-per-task=1
#SBATCH --array=0-1000  # 1001 is upper limit
##SBATCH --array=0-2
#SBATCH --array=0-999
#SBATCH --array=0-1
#SBATCH --array=0-164
#SBATCH --output=./joeis%x%A.out

echo "=============================================="

echo "Starting time (of array_subjob "$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"):"
date

# $1=batch (0-34); $2=dir_id
# pip install pysindy sympy numpy diophantine pandas

milenia=$1
if [[ $1 = "" ]]
then
  milenia=0
fi
echo "input:" $milenia "after"


cd oeis/
#singularity exec ../pg.sif python3 doones.py --job_id $SLURM_ARRAY_JOB_ID \
singularity exec ../oeis.sif python3 doones.py \
        --task_id $(($milenia*1000 + $SLURM_ARRAY_TASK_ID)) --exper_id $SLURM_JOB_NAME

date

echo "this is oei.sh $1 $2 $3 $4 $5 $6 $7 doing \
  doones job_id $SLURM_ARRAY_JOB_ID task_id $SLURM_ARRAY_TASK_ID --exper_id $SLURM_JOB_NAME"
