import math
import torch
import torch.nn as nn


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(0, len(sizes) - 1):
            if i < (len(sizes)-2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


def SumRateLoss(data, output, noise, K=10, binary=False, persample=False):
    bs = data.shape[0]
    H = torch.reshape(data, (bs, K, K))
    H2 = torch.square(H)
    if binary:
        output = torch.round(output)
    pv = torch.reshape(output, (bs, K, 1))
    rx_power = torch.mul(H2, pv)
    mask = torch.eye(K)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), dim=1)
    interference = torch.sum(torch.mul(rx_power, 1-mask), axis=1) + noise
    pyrate = torch.log2(1 + torch.div(valid_rx_power, interference))
    pyrate = torch.sum(pyrate, dim=1)
    if persample:
        return - pyrate
    else:
        return - torch.mean(pyrate)


def weighted_mse_loss(input, target, weight):
    MSE_per_sample = torch.sum((input - target) ** 2, 1)
    return torch.sum(weight * MSE_per_sample)


def weighted_sumrate_loss(data, predict, weight, noise_power):
    SumRate_per_sample = SumRateLoss(
        data, predict, noise_power, persample=True)
    return torch.sum(weight * SumRate_per_sample)


def weighted_ratio_loss(data, predict, target, noise_power, weight=None):
    predict_sumrate = SumRateLoss(data, predict, noise_power, persample=True)
    label_sumrate = SumRateLoss(data, target, noise_power, persample=True)
    ratio = torch.div(predict_sumrate, label_sumrate)
    if weight is None:
        return -ratio
    else:
        return -torch.sum(weight * ratio)


def mse_per_sample(data, predict, target, noise_power=None):
    return torch.sum((predict - target) ** 2, 1)
