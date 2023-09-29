import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import torch.nn.functional as F
import os
import operator
import random
from functools import reduce
from functools import partial
from einops import rearrange
from torch.nn.modules.loss import _WeightedLoss
from dgl.nn.pytorch import SumPooling, AvgPooling



#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random SEED set as {seed}")


#################################################
#################################################
# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        
    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#######################################################################
#######################################################################
# The below loss function is taken from oformer paper to train the oformer model
class FormerLoss(object):
    def __init__(self, res, ed_ratio=0.1):
        super(FormerLoss, self).__init__()
        self.res = res
        self.dx = 1./self.res
        self.ratio = ed_ratio

    def central_diff(self, x: torch.Tensor, h, resolution):
        # assuming PBC
        # x: (batch, n, feats), h is the step size, assuming n = h*w
        x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
        x = F.pad(x,
                (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
        grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2*h)  # f(x+h) - f(x-h) / 2h
        grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2*h)  # f(x+h) - f(x-h) / 2h

        return grad_x, grad_y

    def rel_loss(self, x, y, p, reduction=True, size_average=False, time_average=False):
        # x, y: [b, c, t, h, w] or [b, c, t, n]
        batch_num = x.shape[0]
        frame_num = x.shape[2]

        if len(x.shape) == 5:
            h = x.shape[3]
            w = x.shape[4]
            n = h*w
        else:
            n = x.shape[-1]
        # x = rearrange(x, 'b c t h w -> (b t h w) c')
        # y = rearrange(y, 'b c t h w -> (b t h w) c')
        num_examples = x.shape[0]
        eps = 1e-6
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p, 1) + eps

        loss = torch.sum(diff_norms/y_norms)
        if reduction:
            loss = loss / batch_num
            if size_average:
                loss /= n
            if time_average:
                loss /= frame_num

        return loss


    def rel_l2norm_loss(self, x, y):
        #   x, y [b, c, t, n]
        eps = 1e-6
        y_norm = (y**2).mean(dim=-1) + eps
        diff = ((x-y)**2).mean(dim=-1)
        diff = diff / y_norm   # [b, c, t]
        diff = diff.sqrt().mean()
        return diff

    def pointwise_rel_l2norm_loss(self, x, y):
        #   x, y [b, n, c]
        eps = 1e-6
        y_norm = (y**2).mean(dim=-2) + eps
        diff = ((x-y)**2).mean(dim=-2)
        diff = diff / y_norm   # [b, c]
        diff = diff.sqrt().mean()
        return diff
    
    def __call__(self, input, target):
        pred_loss = self.pointwise_rel_l2norm_loss(input, target)
        gt_grad_x, gt_grad_y = self.central_diff(target, self.dx, self.res)
        pred_grad_x, pred_grad_y = self.central_diff(input, self.dx, self.res)
        deriv_loss = self.pointwise_rel_l2norm_loss(pred_grad_x, gt_grad_x) +\
                        self.pointwise_rel_l2norm_loss(pred_grad_y, gt_grad_y)
        return pred_loss + self.ratio * deriv_loss

class FormerLoss1D(object):
    def __init__(self, res, ed_ratio=0.1):
        super(FormerLoss1D, self).__init__()
        self.res = res
        self.dx = 1./self.res
        self.ratio = ed_ratio

    def central_diff(self, x: torch.Tensor, h):
        # assuming PBC
        # x: (batch, seq_len, feats), h is the step size

        pad_0, pad_1 = x[:, -2:-1], x[:, 1:2]
        x = torch.cat([pad_0, x, pad_1], dim=1)
        x_diff = (x[:, 2:] - x[:, :-2])/2  # f(x+h) - f(x-h) / 2h
        # pad = np.zeros(x_diff.shape[0])

        # return np.c_[pad, x_diff/h, pad]
        return x_diff/h

    def rel_loss(self, x, y, p, reduction=True, size_average=False, time_average=False):
        # x, y: [b, c, t, h, w] or [b, c, t, n]
        batch_num = x.shape[0]
        frame_num = x.shape[2]

        if len(x.shape) == 5:
            h = x.shape[3]
            w = x.shape[4]
            n = h*w
        else:
            n = x.shape[-1]
        # x = rearrange(x, 'b c t h w -> (b t h w) c')
        # y = rearrange(y, 'b c t h w -> (b t h w) c')
        num_examples = x.shape[0]
        eps = 1e-6
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p, 1) + eps

        loss = torch.sum(diff_norms/y_norms)
        if reduction:
            loss = loss / batch_num
            if size_average:
                loss /= n
            if time_average:
                loss /= frame_num

        return loss
    
    def __call__(self, input, target):
        pred_loss = self.rel_loss(input, target, 2)
        gt_grad_x = self.central_diff(target, self.dx)
        pred_grad_x = self.central_diff(input, self.dx)
        deriv_loss = self.rel_loss(pred_grad_x, gt_grad_x, 2)
        return pred_loss + self.ratio * deriv_loss

#loss function with rel/abs Lp loss 
# taken from fno paper
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
class GANLoss():
    def __init__(self, loss=nn.BCELoss()) -> None:
        self.loss = loss
        
    def __call__(self, out):
        return 0.5 * (self.loss(out[0], torch.ones_like(out[0])) + \
                      self.loss(out[1], torch.zeros_like(out[1])))

class EncoderLoss():
    def __init__(self, loss1=nn.BCELoss(), loss2=LpLoss(), ratio=0.05):
        self.loss1 = loss1
        self.loss2 = loss2
        self.ratio = ratio

    def __call__(self, pred, dis_out, y):
        l1 = self.loss1(dis_out, torch.ones_like(dis_out))
        l2 = self.loss2(pred.view(pred.shape[0], -1), y.view(y.shape[0], -1))
        return self.ratio* l1 +  l2, l2

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

#######################################################################
#######################################################################
#taken from gnot paper
class WeightedLpRelLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0,regularizer=False, normalizer=None):
        super(WeightedLpRelLoss, self).__init__()
        self.d = d
        self.p = p
        self.component = component if component == 'all' or 'all-reduce' else int(component)
        self.regularizer = regularizer
        self.normalizer = normalizer
        self.sum_pool = SumPooling()

    ### all reduce is used in temporal cases, use only one metric for all components
    def forward(self, g, pred, target):
        if (self.component == 'all') or (self.component == 'all-reduce'):
            err_pool = (self.sum_pool(g, (pred - target).abs() ** self.p))
            target_pool = (self.sum_pool(g, target.abs() ** self.p))
            losses = (err_pool / target_pool)**(1/ self.p)
        else:
            assert self.component <= target.shape[1]
            err_pool = (self.sum_pool(g, (pred - target[:,self.component]).abs() ** self.p))
            target_pool = (self.sum_pool(g, target[:,self.component].abs() ** self.p))
            losses = (err_pool / target_pool)**(1/ self.p)
        loss = losses.mean()
        return loss