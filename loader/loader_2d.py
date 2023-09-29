import numpy as np
import torch
    
class DataLoader_2D(object):
    def __init__(self, X, y, n, res, grid_size, batch_size, x_normalizer, y_normalizer=None, input_noise=False, input_gamma=0.0, output_noise=False, output_gamma=0.0):
        super(DataLoader_2D, self).__init__()
        self.X = X
        self.y = y
        self.n = n
        self.res = res
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        
        if input_noise:
            noise = torch.var(X, dim=0, keepdim=True)
            self.X += torch.rand(X.shape) * noise * input_gamma
        if output_noise:
            noise = torch.var(y, dim=0, keepdim=True)
            self.y += torch.rand(y.shape) * noise * output_gamma
        
        # Normalizing the Input
        if self.x_normalizer is not None:
            self.X = self.x_normalizer.encode(self.X)
        if self.y_normalizer is not None:
            self.y = self.y_normalizer.encode(self.y)

    def get_grid(self):
        r = (self.grid_size-1) // (self.res-1)
        #Creating the uniform grid for x and y locations of the input
        grids = []
        grid_all = np.linspace(0, 1, self.grid_size).reshape(self.grid_size, 1).astype(np.float64)
        grids.append(grid_all[::r,:])
        grids.append(grid_all[::r,:])
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(self.res,self.res,2)
        grid = torch.tensor(grid, dtype=torch.float)
        return grid

    def get_grid_loader(self, shuffle=False):
        grid = self.get_grid().unsqueeze(0)
        # attaching the grid with the original input
        X_out = torch.cat([self.X.reshape(self.n,self.res,self.res,-1), grid.repeat(self.n,1,1,1)], dim=3)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_out, self.y), batch_size=self.batch_size, shuffle=shuffle)

    def get_loader(self, shuffle=False):
        self.X = self.X.reshape(self.n,self.res,self.res,-1)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.X, self.y), batch_size=self.batch_size, shuffle=shuffle)
