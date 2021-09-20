#!/bin/bash

#SBATCH --job-name=hp # create a short name for your job
#SBATCH --partition=broadwl
#SBATCH --ntasks=15      # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4000
#SBATCH --output=cpu.out
#SBATCH --error=cpu.err
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=broadwl
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=mma3@chicagobooth.edu

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

module unload python
module load python/cpython-3.7.0
module load cuda/11.0
conda init bash
conda activate mlval


for _ in {1..15}; do
python train_qlearner.py --hand_size=5 --nlab1=3 --nlab2=3 --agent_type=Att3 --nepisodes=500000 --batch_size=512 --replay_capacity=200000 --update_frequency=100 &
done