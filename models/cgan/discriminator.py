import torch
import torch.nn as nn

Conv = {
    '1D': nn.Conv1d,
    '2D': nn.Conv2d,
}

Norm = {
    '1D': nn.BatchNorm1d,
    '2D': nn.BatchNorm2d,
}

class ck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, dim=2):
        super().__init__()
        conv = Conv[f'{dim}D']
        norm = Norm[f'{dim}D']
        layer = []
        layer += [conv(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]
        
        if norm != None:
            layer += [norm(out_ch)]
        
        if relu != None:
            layer += [nn.LeakyReLU(relu)]
        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(x)
    
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d=64, norm="bnorm", dim=2):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.disc1 = ck(channels_img  , features_d    , kernel_size=4, stride=2, padding=1, norm=None, relu=0.2, dim=dim)
        self.disc2 = ck(features_d    , features_d * 2, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, dim=dim)
        self.disc3 = ck(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, dim=dim)
        self.disc4 = ck(features_d * 4, features_d * 6, kernel_size=4, stride=1, padding=1, norm=norm, relu=0.2, dim=dim)
        self.disc5 = ck(features_d * 6, 1             , kernel_size=4, stride=1, padding=1, norm=None, relu=None, dim=dim)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):  
        if self.dim == 2:
            x = x.permute(0,3,1,2)  
        else:
            x = x.permute(0,2,1)   
        x = self.disc1(x)    
        x = self.disc2(x) 
        x = self.disc3(x) 
        x = self.disc4(x) 
        x = self.disc5(x)
        x = self.activation(x)
        if self.dim == 2:
            x = x.permute(0,2,3,1)
        else:
            x = x.permute(0,2,1)     
        return x