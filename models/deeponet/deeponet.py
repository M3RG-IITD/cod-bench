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
    def __init__(self, BNET_ARCH=(1, (47,47)), TNET_ARCH=(2,[128,128,128,128],47), POD_BASIS=None, CONST_Y_LOC=None, modes=128):
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
        self.in_channels = BNET_ARCH[0]
        self.data_sz = np.prod(BNET_ARCH[1])
        # self.branch = nn.Sequential(
        #     nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=5, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(in_features=128*self.data_sz, out_features=128),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=modes)
        # )
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=256*self.data_sz, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=modes)
        )

        if self.trunk:
            self.trunk_layer1 = nn.Linear(self.features, self.architecture[0])
            self.trunk_layers = nn.ModuleList([nn.Linear(self.architecture[i], self.architecture[i+1]) for i in range(len(self.architecture)-1)])
            torch.nn.init.xavier_normal_(self.trunk_layer1.weight)
            torch.nn.init.zeros_(self.trunk_layer1.bias)
            for layer in self.trunk_layers:
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.relu = nn.ReLU()
        self.b = torch.tensor(0.0, requires_grad=True)


    def forward(self, x1, x2=None):
        # x2 is the location of points where we wanna estimate the output function value
        # x2 can be none in two cases
        # 1. PODDeepONet is being used and pod_basis is already passed,
        #  and that is the only information we wanna use about the location points of output function
        # 2. DeepONet with a consistent set of grid points for output location is used,
        #  and hence it can be passed as the variable self.grid while creating the model and x2 can be none. 
        # Note it can be done the other way around and grid being passed as x2 and the CONST_Y_LOC being kept as none. 
        x1 = x1.permute(0,3,1,2)
        out_branch = self.branch(x1)
        out = x2 if x2 is not None else self.grid
        if self.trunk:
            out = self.trunk_layer1(out)
            out = self.relu(out)
            for layer in self.trunk_layers:
                out = layer(out)
                out = self.relu(out)
            if self.basis is not None:
                out = torch.concat((self.basis, out), 1)
            out = torch.einsum('bp, xyp -> bxy', out_branch, out)
        else:
            out = torch.einsum('bp, xyp -> bxy', out_branch, self.basis)
        out = out + self.b
        out = out.unsqueeze(-1)
        return out

class PODDeepONet(DeepONet):
    def __init__(self, normalized_y_train, BNET_ARCH=(1, (47,47)), TNET_ARCH=None, modes=115):
        y_mean, v = pod(normalized_y_train.numpy())
        self.y_mean = torch.from_numpy(y_mean).reshape((1,) + BNET_ARCH[1])
        v = torch.tensor(v.copy())
        basis = v[:, :modes].reshape(BNET_ARCH[1] + (modes,))
        super(PODDeepONet, self).__init__(BNET_ARCH=BNET_ARCH, TNET_ARCH=TNET_ARCH, POD_BASIS=basis, modes=modes)   

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