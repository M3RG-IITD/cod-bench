import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

Conv = {
    1: nn.Conv1d,
    2: nn.Conv2d,
}

MaxPool = {
    1: nn.MaxPool1d,
    2: nn.MaxPool2d,
}

ConvTranspose = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
}

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, dim=2):
        super().__init__()
        self.conv1 = Conv[dim](in_ch, out_ch, kernel_size=3,stride = 1, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = Conv[dim](out_ch, out_ch,kernel_size=3,stride = 1, padding=1)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024), dim=2):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1], dim) for i in range(len(chs)-1)])
        self.pool       = MaxPool[dim](2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)         
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), dim=2):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([ConvTranspose[dim](chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1], dim) for i in range(len(chs)-1)]) 
        self.dim = dim
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        if self.dim == 2:
            _, _, H, W = x.shape
            enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        else:
            _, _, target_length = x.size()
            _, _, enc_length = enc_ftrs.size()
            if target_length < enc_length:
                diff = (enc_length - target_length) // 2
                enc_ftrs = enc_ftrs[:, :, diff:diff+target_length]
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, in_channels=3, enc_chs=(64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=True, output_shape=(47,47), dim=2):
        super().__init__()
        self.dim = dim
        enc_chs = (in_channels,) + enc_chs
        self.encoder     = Encoder(enc_chs, dim)
        self.decoder     = Decoder(dec_chs, dim)
        self.head        = Conv[dim](dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.output_shape = output_shape

    def forward(self, x):
        if self.dim == 2:
            x = x.permute(0,3,1,2)
        else:
            x = x.permute(0,2,1)
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.output_shape)
        if self.dim == 2:
            out = out.permute(0,2,3,1)
        else:
            out = out.permute(0,2,1)
        return out