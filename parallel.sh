#!/bin/bash

#SBATCH --job-name=hpp # create a short name for your job
#SBATCH --partition=broadwl
#SBATCH --ntasks=26      # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4000
#SBATCH --output=cpu.out
#SBATCH --error=cpu.err
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=broadwl
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=mma3@chicagobooth.edu

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

module load parallel
module unload python
module load python/cpython-3.7.0
module load cuda/11.0
conda init bash
conda activate mlval

srun="srun --exclusive -N1 -n1"
parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog runtask.log --resume"

$parallel "$srun ./train.sh {1} {2}" ::: {1..13} ::: FF LSTM