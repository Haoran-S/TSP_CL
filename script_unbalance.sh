#!/usr/bin/bash
echo "*********************** generate data ***********************"
python3 generate_data.py --distribution Rayleigh1-Rayleigh5 --num_train 2000-18000  --noise 0.01 --o data/dataset_unbalance.pt

TrainScript="--hidden_layers 200-80-80 --batch_size 500 --noise 0.01 --mini_batch_size 10 --lr 0.001 --n_memories 100 --data_file data/dataset_unbalance.pt --file_ext _unbalance"

echo "*********************** TL ***********************"
python3 main.py $TrainScript --model single

echo "*********************** Joint ***********************"
python3 main.py $TrainScript --model single --mode joint

echo "*********************** Reservoir ***********************"
python3 main.py $TrainScript --model reservoir_sampling

echo "*********************** Minmax (Proposed) ***********************"
python3 main.py $TrainScript --model minmax --dual_stepsize 0.0001

echo "*********************** generate figure ***********************"
python3 generate_figure.py --ext _unbalance


    
        
