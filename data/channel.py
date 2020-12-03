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

import numpy as np
import data.function_wmmse_powercontrol as wf
import re


def generate_geometry_CSI(K, num_H, rng, area_length=10, alpha=2):
    tx_pos = np.zeros([num_H, K, 2])
    rx_pos = np.zeros([num_H, K, 2])
    rayleigh_coeff = np.zeros([num_H, K, K])
    for i in range(num_H):
        tx_pos[i, :, :] = rng.rand(K, 2) * area_length
        rx_pos[i, :, :] = rng.rand(K, 2) * area_length
        rayleigh_coeff[i, :, :] = (
            np.square(rng.randn(K, K)) + np.square(rng.randn(K, K))) / 2

    tx_pos_x = np.reshape(tx_pos[:, :, 0], [num_H, K, 1]) + np.zeros([1, 1, K])
    tx_pos_y = np.reshape(tx_pos[:, :, 1], [num_H, K, 1]) + np.zeros([1, 1, K])
    rx_pos_x = np.reshape(rx_pos[:, :, 0], [num_H, 1, K]) + np.zeros([1, K, 1])
    rx_pos_y = np.reshape(rx_pos[:, :, 1], [num_H, 1, K]) + np.zeros([1, K, 1])
    d = np.sqrt(np.square(tx_pos_x - rx_pos_x) +
                np.square(tx_pos_y - rx_pos_y))
    G = np.divide(1, 1 + d**alpha)
    G = G * rayleigh_coeff
    return np.sqrt(np.reshape(G, [num_H, K ** 2]))


def generate_rayleigh_CSI(K, num_H, rng, diag_ratio=1):
    X = np.zeros((num_H, K ** 2))
    for loop in range(num_H):
        CH = 1 / np.sqrt(2) * (rng.randn(K, K) +
                               1j * rng.randn(K, K))
        CH = CH + (diag_ratio - 1) * np.diag(np.diag(CH))
        CH = np.reshape(CH, (1, K**2))
        X[loop, :] = abs(CH)
    return X


def generate_rice_CSI(K, num_H, rng):
    X = np.zeros((num_H, K ** 2))
    for loop in range(num_H):
        CH = 1 / 2 * (1 + rng.randn(1, K ** 2) +
                      1j * (1 + rng.randn(1, K ** 2)))
        X[loop, :] = abs(CH)
    return X


def generate_CSI(K, num_H, seed, distribution, var_noise):
    rng = np.random.RandomState(seed)

    out = re.split(r'(\d+)', distribution)
    name, num = (out[0], 1) if len(out) == 1 else (out[0], out[1])
    print('Generate Data: %s, %s, K = %d, num = %d, seed = %d' %
          (name, num, K, num_H, seed))

    if name == "Rayleigh":
        abs_H = generate_rayleigh_CSI(K, num_H, rng, int(num))
    elif name == "Rice":
        abs_H = generate_rice_CSI(K, num_H, rng)
    elif name == "Geometry":
        abs_H = generate_geometry_CSI(K, num_H, rng, area_length=int(num))
    else:
        print("Invalid CSI distribution.")
        exit(0)

    Pmax = 1
    Pini = np.ones(K)
    Y = np.zeros((num_H, K))
    for loop in range(num_H):
        H = np.reshape(abs_H[loop, :], (K, K))
        Y[loop, :] = wf.WMMSE_sum_rate(Pini, H, Pmax, var_noise)
    return abs_H, Y
