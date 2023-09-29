"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)


""" Def: 2d Wavelet layer """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, shape, device='cuda'):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.dwt = DWT(J=self.level, mode='symmetric', wave='db4').to(device)
        self.idwt = IDWT(mode='symmetric', wave='db4').to(device)
        self.mode_data, _ = self.dwt(torch.rand((1,in_channels)+shape).to(device))
        self.modes1 = self.mode_data.shape[-2]
        self.modes2 = self.mode_data.shape[-1]
        self.device = device
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet
        x_ft, x_coeff = self.dwt(x)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batch_size, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=self.device)
        out_ft = self.mul2d(x_ft, self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        x_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        x_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space        
        x = self.idwt((out_ft, x_coeff))
        return x

""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width=64, level=4, shape=(47,47), in_channels=3, device='cuda'):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.level = level
        self.width = width
        self.padding = 1 # pad the domain when required
        self.fc0 = nn.Linear(in_channels, self.width) # input channel is 3: (a(x, y), x, y)
        self.pad = True
        if shape[0]%2==0:
            self.pad = False

        self.conv0 = WaveConv2d(self.width, self.width, self.level, shape, device)
        self.conv1 = WaveConv2d(self.width, self.width, self.level, shape, device)
        self.conv2 = WaveConv2d(self.width, self.width, self.level, shape, device)
        self.conv3 = WaveConv2d(self.width, self.width, self.level, shape, device)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 192)
        self.fc2 = nn.Linear(192, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        if self.pad:
            x = F.pad(x, [0,self.padding, 0,self.padding])

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

        if self.pad:
            x = x[..., :-self.padding, :-self.padding] 
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
