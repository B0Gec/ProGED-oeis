#!/bin/sh
#SBATCH --job-name=array_test
#SBATCH --time=00:10:00
#SBATCH --output=array_test-%a.txt   # %a je nastavek za oznako naloge
#SBATCH --array=0-3               # območje spreminjanja vrednosti

for i in {0..2} 
do
	{
	echo tro $i
	echo slurm $(($SLURM_ARRAY_TASK_ID+$i))
	timedatectl | grep Local;
	sleep 5
	echo "to" $i
	timedatectl | grep Local;
	} &
done
wait
echo "done"
