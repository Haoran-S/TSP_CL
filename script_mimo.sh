#!/usr/bin/bash
TrainScript="--hidden_layers 200-80-80 --batch_size 5000 --noise 1e-12 --n_iter 10  --lr 0.001 --mini_batch_size 100 --n_memories 2000 --data_file data/dataset_deepmimo_fastx3.pt --file_ext _mimo"

#TrainScript="--hidden_layers 200-80-80 --batch_size 5000 --noise 1e-12 --n_iter 10  --lr 0.001 --mini_batch_size 100 --n_memories 2000 --data_file data/dataset_deepmimo_slow.pt --file_ext _mimo"


echo "*********************** TL ***********************"
python3 main.py $TrainScript --model single

echo "*********************** Reservoir ***********************"
python3 main.py $TrainScript --model reservoir_sampling

echo "*********************** Minmax (Proposed) ***********************"
python3 main.py $TrainScript --model minmax --dual_stepsize 0.00001

echo "*********************** Compositionl (Proposed) ***********************"
python3 main.py $TrainScript --model composition

echo "*********************** Joint ***********************"
python3 main.py $TrainScript --model single --mode joint

echo "*********************** Joint (Compositionl) ***********************"
python3 main.py $TrainScript --model composition --mode joint

echo "*********************** Generate Figure ***********************"
python3 generate_figure.py --ext _mimo
