import torch
import torch.nn as nn

class ck2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0):
        super().__init__()
        layer = []
        layer += [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]
        
        if norm != None:
            layer += [nn.BatchNorm2d(out_ch)]
        
        if relu != None:
            layer += [nn.LeakyReLU(relu)]
        
        self.net = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.net(x)
    
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d=64, norm="bnorm"):
        super(Discriminator, self).__init__()
    
        self.disc1 = ck2d(channels_img  , features_d    , kernel_size=4, stride=2, padding=1, norm=None, relu=0.2)
        self.disc2 = ck2d(features_d    , features_d * 2, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2)
        self.disc3 = ck2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2)
        self.disc4 = ck2d(features_d * 4, features_d * 6, kernel_size=4, stride=1, padding=1, norm=norm, relu=0.2)
        self.disc5 = ck2d(features_d * 6, 1             , kernel_size=4, stride=1, padding=1, norm=None, relu=None)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):  
        x = x.permute(0,3,1,2)     
        x = self.disc1(x)    
        x = self.disc2(x) 
        x = self.disc3(x) 
        x = self.disc4(x) 
        x = self.disc5(x)
        x = self.activation(x)
        x = x.permute(0,2,3,1)     
        return x