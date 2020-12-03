#!/usr/bin/bash
echo "*********************** generate data ***********************"
python3 generate_data.py --distribution Rayleigh-Rice-Geometry10-Geometry50 --num_train 20000-20000-20000-20000 --noise 1.0 --o data/dataset_balance.pt

TrainScript="--hidden_layers 200-80-80 --batch_size 5000 --noise 1.0 --mini_batch_size 100 --lr 0.001 --n_memories 2000 --data_file data/dataset_balance.pt --file_ext _balance"

echo "*********************** TL ***********************"
python3 main.py $TrainScript --model single

echo "*********************** Joint ***********************"
python3 main.py $TrainScript --model single --mode joint

echo "*********************** Reservoir ***********************"
python3 main.py $TrainScript --model reservoir_sampling

echo "*********************** Minmax (Proposed) ***********************"
python3 main.py $TrainScript --model minmax --dual_stepsize 0.0001

echo "*********************** generate figure ***********************"
python3 generate_figure.py --ext _balance
