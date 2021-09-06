#!/bin/bash
job_id=$(sbatch tensorboard.bash $1 | awk '{print $4}')
sleep 4
echo "Job $job_id"
cat "slurm-$job_id.out"
rm "slurm-$job_id.out"