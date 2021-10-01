#!/bin/bash

module load parallel
module unload python
module load python/cpython-3.7.0
module load cuda/11.0
conda init bash
conda activate mlval


echo task $1 $2 seq:$PARALLEL_SEQ host:$(hostname) date:$(date)

python train_qlearner.py --hand_size=5 --nlab1=3 --nlab2=3 --agent_type=$2 --nepisodes=4000000 --batch_size=500 --replay_capacity=300000 --update_frequency=500 



