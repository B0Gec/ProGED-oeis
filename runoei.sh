#!/bin/bash

date

for i in {0..35}
do
	{
  sbatch oei.sh $i $1 $2 $3 $4 $5
	}	&
done
wait
date

touch `date "+%H-%M-%S"`$0.txt

echo "this is $0 $1 $2 $3 $4 $5 $6 $7"
