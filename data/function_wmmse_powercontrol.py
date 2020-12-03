# ###############################################
# This file was written for ``Learning to Continuously Optimize" [1].
# Originally forked from ``Learning to Optimize" [2].
# Also includes functions to perform the WMMSE algorithm [3].
# Codes have been tested successfully on Python 3.6.0.
#
# References:
# [1] Haoran Sun, Wenqiang Pu, Minghe Zhu, Xiao Fu, Tsung-Hui Chang,
# Mingyi Hong, "Learning to Continuously Optimize Wireless Resource In
# Episodically Dynamic Environment",
# arXiv preprint arXiv:2011.07782 (2020).
#
# [2] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong,
# Xiao Fu and Nikos D. Sidiropoulos, “Learning to Optimize: Training Deep
# Neural Networks for Wireless Resource Management”, IEEE Transactions on
# Signal Processing 66.20 (2018): 5438-5453.
#
# [3] Qingjiang Shi, Meisam Razaviyayn, Zhi-Quan Luo, and Chen He.
# "An iteratively weighted MMSE approach to distributed sum-utility
# maximization for a MIMO interfering broadcast channel."
# IEEE Transactions on Signal Processing 59.9 (2011): 4331-4340.
#
# version 1.0 -- February 2017.
# Written by Haoran Sun (sunhr1993 @ gmail.com)
# All rights reserved.
# ###############################################

import numpy as np
import math


def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    """Functions for WMMSE algorithm"""
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / \
                sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / \
                ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            vnew = vnew + math.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-3:
            break

    p_opt = np.square(b)
    return p_opt
