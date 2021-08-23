#!/bin/bash

#SBATCH --job-name=gpu   # job name
#SBATCH --output=gpu.out # output log file
#SBATCH --error=gpu.err  # error file
#SBATCH --time=01:00:00  # 1 hour of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=4       # 4 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU


echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

module unload python
module load python/cpython-3.7.0
module load cuda/11.0
conda init bash
conda activate mlval


python train_qlearner.py