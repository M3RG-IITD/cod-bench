"""
This file is based on Spectral Neural Operator defined in paper [https://arxiv.org/abs/2205.10573.pdf].

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .activations import *

def align_frequencies(values, forward=True):
    # itorchut array has shape `(n1, n2, ..., c)`, where c stands for channels
    # if `forward` is `True` reorder elements according to the increase in magnitude of frequency
    # if `forward` is `False` perform inverse transposition to the case when forward is `True`
    D = len(values.shape) - 1
    if D == 1:
        return values
    else:
        transposition = [i for i in range(D-2)] + [i for i in range(D-1, D+1)]
        transposition = [D-2,] + transposition if D > 1 else transposition
    for i in range(D-1):
        order = torch.argsort(abs(torch.fft.fftfreq(len(values))))
        if not forward:
            order = torch.tensor([x[0] for x in sorted(enumerate(order), key=lambda x: x[1])], dtype=torch.int64)
        values = torch.permute(values[order], dims=transposition)
    return values

def reweight(values, device):
    # fix signs to comply with Fourier series
    D = len(values.shape)
    for i, s in enumerate(values.shape[:-1]):
        weight = (-1)**torch.arange(s, dtype=int, device=device)
        values = values*weight.reshape([1 if j != i else s for j in range(D)])
    return values

def in_transform(values, device):
    # itorchut array has shape `(n1, n2, ..., c)`, where c stands for channels
    # transform values of the function on the uniform grid to coefficients of trigonometric series
    # coefficients are reordered such that larger index corresponds to larger frequency
    coeff = torch.fft.rfftn(values, dim=[i for i in range(len(values.shape)-1)], norm="forward")
    coeff = reweight(coeff, device)
    coeff = align_frequencies(coeff).to(device)
    return coeff

def out_transform(coefficients, shape, device):
    # x = coefficients_to_values(values_to_coefficients(x), x.shape) up to round-off error
    coeff = align_frequencies(coefficients, forward=False).to(device)
    coeff = reweight(coeff, device)
    values = torch.fft.irfftn(coeff, s=shape[:-1], dim=[i for i in range(len(coefficients.shape)-1)], norm="forward")
    return values.to(device)

class NN_c(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU()):
        super(NN_c, self).__init__()
        self.params = NN_c.init_c_network_params(sizes)
        self.activation = activation

    def np_c_layer_params(m, n):
        layer_parameters = nn.ParameterList([nn.Parameter(torch.normal(mean=torch.zeros(m, n)) / m), nn.Parameter(torch.normal(mean=torch.zeros(m, n)) / m), nn.Parameter(torch.normal(mean=torch.zeros(n))), nn.Parameter(torch.normal(mean=torch.zeros(1, n)))])
        return layer_parameters
    
    def init_c_network_params(sizes):
        return nn.ParameterList([NN_c.np_c_layer_params(m, n) for m, n in zip(sizes[:-1], sizes[1:])])
    
    def forward(self, U):
        n = len(self.params)
        for i, p in enumerate(self.params):
            A = p[0] + 1j * p[1]
            b = p[2] + 1j * p[3]
            # shape of A {c_m,c_n} and shape of U {b,x,c_m} -> resulting shape of U: {b,x,c_n}
            U = torch.einsum('bxc, cd -> bxd', U , A)
            # shape of b {1,c_n} and shape of U {b,x,c_n} -> resulting shape of U: {b,x,c_n}
            U += b
            if i < n-1:
                U = self.activation(U)
        return U
    
class NN_i(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU()):
        super(NN_i, self).__init__()
        self.params = NN_i.init_i_network_params(sizes)
        self.activation = activation

    def np_i_layer_params(m_x, n_x, c_m, c_n):
        layer_parameters = nn.ParameterList([nn.Parameter(torch.normal(mean=torch.zeros(n_x, m_x)) / m_x), nn.Parameter(torch.normal(mean=torch.zeros(n_x, m_x)) / m_x), nn.Parameter(torch.normal(mean=torch.zeros(c_m, c_n)) / c_m), nn.Parameter(torch.normal(mean=torch.zeros(c_m, c_n)) / c_m), nn.Parameter(torch.normal(mean=torch.zeros(n_x, c_n))), nn.Parameter(torch.normal(mean=torch.zeros(n_x, c_n)))])
        return layer_parameters

    def init_i_network_params(sizes):
        return nn.ParameterList([NN_i.np_i_layer_params(m_x, n_x, c_m, c_n) for m_x, n_x, c_m, c_n in zip(sizes[0][:-1], sizes[0][1:], sizes[1][:-1], sizes[1][1:])])
    
    def forward(self, U):
        n = len(self.params)
        for i, p in enumerate(self.params):
            A = p[2] + 1j * p[3]
            B = p[0] + 1j * p[1]
            b = p[4] + 1j * p[5]
            # shape of A {c_m,c_n} and shape of U {b,m_x,c_m} -> resulting shape of U: {b,m_x,c_n}
            U = torch.einsum('bxc, cd -> bxd', U, A)
            # shape of B1 {n_x,m_x} and shape of U {b,m_x,c_n} -> resulting shape of U: {b,n_x,c_n}
            U = torch.einsum('yx, bxc -> byc', B, U)
            # shape of b {n_x,c_n} and shape of U {b,n_x,c_n} -> resulting shape of U: {b,n_x,c_n}
            U += b
            if i < n-1:
                U = self.activation(U)
        return U


class fSNO1d(nn.Module):
    def __init__(self, sizes=([1, 10, 10], [(4097, 5001, 5001, 4097), (10, 10, 10, 10)], [10, 1]),
                out_shape=[8192,1], activation=complex_split_softplus):
        super(fSNO1d, self).__init__()
        self.layer1 = NN_c(sizes[0], activation)
        self.layer2 = NN_i(sizes[1], activation)
        self.layer3 = NN_c(sizes[2], activation)
        self.out_shape = out_shape
        self.device = torch.device('cpu')

    def to(self, device):
        super(fSNO1d, self).to(device=device)
        self.device = device
        return self

    def cuda(self):
        super(fSNO1d, self).cuda()
        self.device = torch.device('cuda')
        return self

    def forward(self, input):
        batch_size = input.shape[0]
        input = in_transform(input, self.device)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = out_transform(input, [batch_size]+self.out_shape, self.device)
        return input.contiguous()
