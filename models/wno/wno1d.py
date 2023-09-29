"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for 1-D Burger's equation (time-independent problem).
"""

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1D, IDWT1D

""" Def: 1d Wavelet layer """
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, shape, device='cuda'):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level 
        self.device = device 
        self.dwt_ = DWT1D(wave='db6', J=self.level, mode='symmetric').to(self.device)
        self.mode_data, _ = self.dwt_(torch.rand((1,1)+shape).to(device))
        self.modes1 = self.mode_data.shape[-1]
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    # Convolution
    def mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet     
        dwt = DWT1D(wave='db6', J=self.level, mode='symmetric').to(self.device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply the final low pass and high pass coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1],  device=self.device)
        out_ft = self.mul1d(x_ft, self.weights1)
        x_coeff[-1] = self.mul1d(x_coeff[-1], self.weights2)
        
        # Reconstruct the signal
        idwt = IDWT1D(wave='db6', mode='symmetric').to(self.device)
        x = idwt((out_ft, x_coeff))        
        return x

""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width=64, level=8, shape=(8192,), in_channels=2, device='cuda'):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.level = level
        self.width = width
        self.padding = 2 # pad the domain when required
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = WaveConv1d(self.width, self.width, self.level, shape, device)
        self.conv1 = WaveConv1d(self.width, self.width, self.level, shape, device)
        self.conv2 = WaveConv1d(self.width, self.width, self.level, shape, device)
        self.conv3 = WaveConv1d(self.width, self.width, self.level, shape, device)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) 

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] 
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

