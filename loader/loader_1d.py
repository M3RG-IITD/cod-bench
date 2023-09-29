import numpy as np
import torch
    
class DataLoader_1D(object):
    def __init__(self, X, y, n, res, grid_size, batch_size, x_normalizer=None, y_normalizer=None):
        super(DataLoader_1D, self).__init__()
        self.X = X
        self.y = y
        self.n = n
        self.res = res
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

    def get_grid(self):
        r = (self.grid_size-1) // (self.res-1)
        #Creating the uniform grid
        grid = np.linspace(0, 1, self.grid_size).reshape(self.grid_size, 1).astype(np.float64)
        grid = grid[::r,:]
        grid = torch.tensor(grid, dtype=torch.float)
        return grid

    def get_grid_loader(self, shuffle=False):
        # Normalizing the Input
        if self.x_normalizer is not None:
            self.X = self.x_normalizer.encode(self.X)
        if self.y_normalizer is not None:
            self.y = self.y_normalizer.encode(self.y)
        grid = self.get_grid().unsqueeze(0)
        # attaching the grid with the original input
        X_out = torch.cat([self.X.reshape(self.n,self.res,1), grid.repeat(self.n,1,1)], dim=2)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_out, self.y), batch_size=self.batch_size, shuffle=shuffle)

    def get_loader(self, shuffle=False):
        # Normalizing the Input
        if self.x_normalizer is not None:
            self.X = self.x_normalizer.encode(self.X)
        if self.y_normalizer is not None:
            self.y = self.y_normalizer.encode(self.y)
        self.X = self.X.reshape(self.n,self.res,1)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.X, self.y), batch_size=self.batch_size, shuffle=shuffle)
