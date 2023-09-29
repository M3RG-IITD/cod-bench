import torch
import torch.nn.functional as F
from einops import rearrange

class OutTransforms():
    def __init__(self, y_normalizer=None, res=None, modes=None, y_mean=None, device='cuda'):         
        self.y_normalizer= y_normalizer
        self.res = res
        self.device = device
        self.modes = modes
        self.y_mean = y_mean

    def stdTransform(self, out):
        self.y_normalizer.to(self.device)
        x_out = self.y_normalizer.decode(out.squeeze(-1)).unsqueeze(-1)
        return x_out
    
    def oformerTransform(self, out):
        if self.y_normalizer is not None:
            self.y_normalizer.to(self.device)
            x_out = self.y_normalizer.decode(out.reshape(out.shape[0],self.res,self.res)).unsqueeze(-1)
        else:
            x_out = out.reshape(out.shape[0],self.res,self.res,1)
        x_out = rearrange(x_out, 'b h w c -> b c h w', h=self.res)
        x_out = x_out[..., 1:-1, 1:-1].contiguous()
        x_out = F.pad(x_out, (1, 1, 1, 1), "constant", 0)
        x_out = rearrange(x_out, 'b c h w -> b (h w) c')
        return x_out
    

    def oformerTransform1D(self, out):
        self.y_normalizer.to(self.device)
        x_out = self.y_normalizer.decode(out.reshape(out.shape[0],self.res)).unsqueeze(-1)
        # x_out = rearrange(x_out, 'b l c -> b c l', l=self.res)
        # x_out = x_out[..., 1:-1].contiguous()
        # # x_out = F.pad(x_out, (1, 1, 1), "constant", 0)
        # x_out = rearrange(x_out, 'b c l -> b l c')
        return x_out
    
    def podnetTransform(self, out):
        if self.y_normalizer is not None:
            self.y_normalizer.to(self.device)
        x_out = out.squeeze(-1)/self.modes
        x_out = torch.add(x_out, self.y_mean.reshape(1, x_out.shape[1], x_out.shape[2]))
        if self.y_normalizer is not None:
            x_out = self.y_normalizer.decode(x_out)
        return x_out.unsqueeze(-1)
    
class InTransforms():
    def __init__(self, grid=None, device='cuda'):
        self.grid = grid
        self.device = device

    def oformerEncoderTransform(self, x, y):
        input_pos = self.grid.repeat([x.shape[0], 1, 1]).to(self.device)
        x = rearrange(x, 'b h w c -> b (h w) c')
        y = rearrange(y.unsqueeze(-1), 'b h w c -> b (h w) c')
        return [x, input_pos], y
    
    def oformerEncoderTransform1D(self, x, y):
        input_pos = self.grid.repeat([x.shape[0], 1, 1]).to(self.device)
        # x = rearrange(x, 'b h w c -> b (h w) c')
        # y = rearrange(y.unsqueeze(-1), 'b h w c -> b (h w) c')
        return [x, input_pos], y.unsqueeze(-1)
    
    
    def oformerDecoderTransform(self, x):
        input_pos = self.grid.repeat([x.shape[0], 1, 1]).to(self.device)
        prop_pos = self.grid.repeat([x.shape[0], 1, 1]).to(self.device)
        return [x, prop_pos, input_pos]
    
    def cganDecoderTransform(self, pred, x, y):
        return torch.cat((x, y.unsqueeze(-1)), dim=-1), torch.cat((x, pred), dim=-1)
    
    

        