# ###############################################
# This file was written for ``Learning to Continuously Optimize" [1].
# DNN model part is modified from ``Learning to Optimize" [2].
# Codes have been tested successfully on Python 3.6.0.
#
# References:
# [1] Haoran Sun, Wenqiang Pu, Minghe Zhu, Xiao Fu, Tsung-Hui Chang,
# Mingyi Hong, "Learning to Continuously Optimize Wireless Resource In
# Episodically Dynamic Environment",
# arXiv preprint arXiv:2011.07782 (2020).
#
# [2] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong, Xiao Fu
# and Nikos D. Sidiropoulos, “Learning to Optimize: Training Deep
# Neural Networks for Wireless Resource Management”,
# IEEE Transactions on Signal Processing 66.20 (2018): 5438-5453.
#
# version 1.0 -- Oct. 2020.
# Haoran Sun (sunhr1993 @ gmail.com)
# All rights reserved.
# ###############################################

import argparse
import torch
from data.channel import generate_CSI


def load_datasets(args):
    d_tr, d_te, args = torch.load(args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = d_tr[0][2].size(1)
    print(args)
    return d_tr, d_te, n_inputs, n_outputs, len(d_tr)


if __name__ == "__main__":
    distribution = "Rayleigh-Rice-Geometry10-Geometry50"
    num_train = "20000"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--o', default='data/dataset_balance.pt', help='output file')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--distribution', default=distribution, type=str)
    parser.add_argument('--noise', default=1.0, type=float)
    parser.add_argument('--num_train', default=num_train, type=str)
    parser.add_argument('--num_test', default=1000, type=int)
    parser.add_argument('--K', default=10, type=int, help='number of user')
    args = parser.parse_args()

    tasks_tr = []
    tasks_te = []
    train_size = [int(k) for k in args.num_train.split('-')]
    data_distribution = args.distribution.split('-')
    assert len(train_size) == 1 or len(train_size) == len(
        data_distribution), "len mismatch"
    for t in range(len(data_distribution)):
        dist = data_distribution[t]
        num_train = train_size[0] if len(train_size) == 1 else train_size[t]

        Xtrain, Ytrain = generate_CSI(
            args.K, num_train, args.seed, dist, args.noise)
        Xtrain = torch.from_numpy(Xtrain).float()
        Ytrain = torch.from_numpy(Ytrain).float()
        tasks_tr.append([dist, Xtrain.clone(), Ytrain.clone()])

        Xtest, Ytest = generate_CSI(
            args.K, args.num_test, args.seed+2020, dist, args.noise)
        Xtest = torch.from_numpy(Xtest).float()
        Ytest = torch.from_numpy(Ytest).float()
        tasks_te.append([dist, Xtest.clone(), Ytest.clone()])

    torch.save([tasks_tr, tasks_te, args], args.o)
