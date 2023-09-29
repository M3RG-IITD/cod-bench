import torch
import torch.nn as nn
import numpy as np

BatchNorm = {
    '1D': nn.BatchNorm1d,
    '2D': nn.BatchNorm2d,
}

Conv = {
    '1D': nn.Conv1d,
    '2D': nn.Conv2d,
}

MaxPool = {
    '1D': nn.MaxPool1d,
    '2D': nn.MaxPool2d,
}

AvgPool = {
    '1D': nn.AdaptiveAvgPool1d,
    '2D': nn.AdaptiveAvgPool2d,
}

class Block(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, conv, norm, stride=1, expansion=4, id_downsample=None) -> None:
        super(Block, self).__init__()
        expansion = expansion
        self.downsample = id_downsample
        self.stride = stride

        self.relu = nn.ReLU()

        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        # self.conv3 = nn.Conv2d(out_channels, out_channels*expansion, kernel_size=1, stride=1, padding=0)
        # self.batch_norm3 = nn.BatchNorm2d(out_channels*expansion)
        self.conv1 = conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = norm(out_channels)
        
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = norm(out_channels)
        
        self.conv3 = conv(out_channels, out_channels*expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = norm(out_channels*expansion)


    def forward(self, x):

        identity = x.clone()
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x))+identity)

        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, out_shape, conv, maxpool, norm, avgpool, num_channels=3, dim=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.dim = dim
        # self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = conv(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm1 = norm(64)
        self.relu = nn.ReLU()
        # self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        self.max_pool = maxpool(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64, conv=conv, norm=norm)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, conv=conv, norm=norm, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, conv=conv, norm=norm, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, conv=conv, norm=norm, stride=2)
        
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool = avgpool(1)
        self.fc = nn.Linear(512*ResBlock.expansion, np.prod(out_shape))
        self.out_shape = out_shape
        
    def forward(self, x):
        if self.dim == 1:
            x = x.permute(0,2,1)
        else:
            x = x.permute(0,3,1,2)

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = x.reshape((-1,) + self.out_shape)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, conv, norm, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                # nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                # nn.BatchNorm2d(planes*ResBlock.expansion)
                conv(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                norm(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, id_downsample=ii_downsample, stride=stride, conv=conv, norm=norm))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes, conv, norm))
            
        return nn.Sequential(*layers)
    

def ResNet50(out_shape:tuple, channels=3, dim=2):
    conv = Conv[f'{dim}D']
    maxpool = MaxPool[f'{dim}D']
    norm = BatchNorm[f'{dim}D']
    avgpool = AvgPool[f'{dim}D']
    return ResNet(Block, [3,4,6,3], out_shape, conv, maxpool, norm, avgpool, channels, dim)