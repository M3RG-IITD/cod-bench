import torch
import torch.nn as nn
import numpy as np

def pod(y):
    batch_size = y.shape[0]
    y = y.reshape(batch_size, -1)
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    C = 1 / (len(y) - 1) * y.T @ y
    w, v = np.linalg.eigh(C)
    w = np.flip(w)
    v = np.fliplr(v)
    v *= len(y_mean) ** 0.5
    return y_mean, v

class DeepONet(nn.Module):
    def __init__(self, in_channels=8192, TNET_ARCH=(1,[2048,2048,2048,45],8192), POD_BASIS=None, CONST_Y_LOC=None, modes=45):
        super().__init__()
        self.basis = POD_BASIS
        # basis = pod_basis in case of PODDeepONet
        self.grid = CONST_Y_LOC
        # grid = output grid location in case of DeepONet with fixed output grid points
        self.trunk = False
        if TNET_ARCH is not None:
            self.trunk = True
            self.features = TNET_ARCH[0]  
            self.architecture = TNET_ARCH[1]  ## LAYER FEATURES
            self.dim = TNET_ARCH[2]
        self.in_channels = in_channels
        self.branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_channels, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.Tanh(),
            nn.Linear(2048, modes),
            nn.Tanh(),
        )

        if self.trunk:
            self.trunk_layer1 = nn.Linear(self.features, self.architecture[0])
            self.trunk_layers = nn.ModuleList([nn.Linear(self.architecture[i], self.architecture[i+1]) for i in range(len(self.architecture)-1)])
            torch.nn.init.xavier_normal_(self.trunk_layer1.weight)
            torch.nn.init.zeros_(self.trunk_layer1.bias)
            for layer in self.trunk_layers:
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.act = nn.Tanh()
        self.b = torch.tensor(0.0, requires_grad=True)


    def forward(self, x1, x2=None):
        # x2 is the location of points where we wanna estimate the output function value
        # x2 can be none in two cases
        # 1. PODDeepONet is being used and pod_basis is already passed,
        #  and that is the only information we wanna use about the location points of output function
        # 2. DeepONet with a consistent set of grid points for output location is used,
        #  and hence it can be passed as the variable self.grid while creating the model and x2 can be none. 
        # Note it can be done the other way around and grid being passed as x2 and the CONST_Y_LOC being kept as none. 
        out_branch = self.branch(x1.unsqueeze(-1)).reshape(x1.shape[0], -1)
        out = x2 if x2 is not None else self.grid
        if self.trunk:
            out = self.trunk_layer1(out)
            for layer in self.trunk_layers:
                out = self.act(out)
                out = layer(out)
            if self.basis is not None:
                out = torch.concat((self.basis, out), 1)
            out = torch.einsum('bp, rp -> br', out_branch, out)
        else:
            out = torch.einsum('bp, rp -> br', out_branch, self.basis)
        out = out + self.b
        out = out.unsqueeze(-1)
        return out

class PODDeepONet(DeepONet):
    def __init__(self, normalized_y_train, in_channels=8192, TNET_ARCH=None, res=8192, modes=32):
        y_mean, v = pod(normalized_y_train.numpy())
        self.y_mean = torch.from_numpy(y_mean).reshape((1, res))
        v = torch.tensor(v.copy())
        basis = v[:, :modes].reshape((res, modes))
        super(PODDeepONet, self).__init__(in_channels=in_channels, TNET_ARCH=TNET_ARCH, POD_BASIS=basis, modes=modes)   

    def __getMean__(self):
        return self.y_mean
    
    def cuda(self):
        super(PODDeepONet, self).cuda()
        self.y_mean = self.y_mean.cuda()
        self.basis = self.basis.cuda()
        return self
    
    def to(self, device):
        super(PODDeepONet, self).to(device)
        self.y_mean = self.y_mean.to(device)
        self.basis = self.basis.to(device)
        return self