"""
This file is based on General Neural Operator Transformer defined in paper [https://arxiv.org/abs/2302.14376.pdf].

"""

from .loader_utils import TorchQuantileTransformer, UnitTransformer, PointWiseUnitTransformer, MultipleTensors
from sklearn.preprocessing import QuantileTransformer
from dgl.data import DGLDataset
import torch
import dgl
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class GNOTDataset(DGLDataset):
    def __init__(self, x, y, n, res, grid_size, dim=2, x_normalizer=None, y_normalizer=None, input_noise=False, output_noise=False, input_gamma=0.0, output_gamma=0.0):
        self.x_data = x.reshape(n, x.shape[1] * x.shape[2], -1)
        self.y_data = y.reshape(n, y.shape[1] * y.shape[2], -1)
        self.n = n
        self.res = res
        self.grid_size = grid_size
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

        if input_noise:
            noise = torch.var(self.x_data, dim=0, keepdim=True)
            self.x_data += torch.rand(self.x_data.shape) * noise * input_gamma
        if output_noise:
            noise = torch.var(self.y_data, dim=0, keepdim=True)
            self.y_data += torch.rand(self.y_data.shape) * noise * output_gamma
        if dim==2:
            self.grid = self.get_grid_2d()
        else:
            self.grid = self.get_grid_1d()
        super(GNOTDataset, self).__init__(' ')

    def get_grid_1d(self):
        r = (self.grid_size-1) // (self.res-1)
        #Creating the uniform grid
        grid = np.linspace(0, 1, self.grid_size).reshape(self.grid_size, 1).astype(np.float64)
        grid = grid[::r,:]
        grid = torch.tensor(grid, dtype=torch.float)
        return grid


    def get_grid_2d(self):
        r = (self.grid_size-1) // (self.res-1)
        #Creating the uniform grid for x and y locations of the input
        grids = []
        grid_all = np.linspace(0, 1, self.grid_size).reshape(self.grid_size, 1).astype(np.float64)
        grids.append(grid_all[::r,:])
        grids.append(grid_all[::r,:])
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(self.res*self.res,2)
        grid = torch.tensor(grid, dtype=torch.float)
        return grid

    def process(self):
        if self.x_normalizer is not None:
            self.x_data = self.x_normalizer.encode(self.x_data)
        if self.y_normalizer is not None:
            self.y_data = self.y_normalizer.encode(self.y_data)
        grid = self.grid.unsqueeze(0)
        # attaching the grid with the original input
        self.x_data = torch.cat([self.x_data, grid.repeat(self.n,1,1)], dim=2)        
        self.data_len = len(self.x_data)
        self.n_dim = self.x_data.shape[1]
        self.graphs = []
        self.graphs_u = []
        self.u_p = []
        for i in range(len(self)):
            x_t, y_t = self.x_data[i].float(), self.y_data[i].float()
            g = dgl.DGLGraph()
            g.add_nodes(self.n_dim)
            g.ndata['x'] = x_t
            g.ndata['y'] = y_t
            up = torch.zeros([1])
            u = torch.zeros([1])
            u_flag = torch.zeros(g.number_of_nodes(),1)
            g.ndata['u_flag'] = u_flag
            self.graphs.append(g)
            self.u_p.append(up) # global input parameters
            g_u = dgl.DGLGraph()
            g_u.add_nodes(self.n_dim)
            g_u.ndata['x'] = x_t
            g_u.ndata['u'] = torch.zeros(g_u.number_of_nodes(), 1)
            self.graphs_u.append(g_u)
        self.u_p = torch.stack(self.u_p)
        return
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.graphs[idx], self.u_p[idx], self.graphs_u[idx]
    

class GNOTDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1,sort_data=True, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        super(GNOTDataLoader, self).__init__(dataset=dataset, batch_size=batch_size,
                                           shuffle=shuffle, sampler=sampler,
                                           batch_sampler=batch_sampler,
                                           num_workers=num_workers,
                                           collate_fn=collate_fn,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last, timeout=timeout,
                                           worker_init_fn=worker_init_fn)

        self.sort_data = sort_data
        if sort_data:
            self.batch_indices = [list(range(i, min(i+batch_size, len(dataset)))) for i in range(0, len(dataset), batch_size)]
            if drop_last:
                self.batch_indices = self.batch_indices[:-1]
        else:
            self.batch_indices = list(range(0, (len(dataset) // batch_size)*batch_size)) if drop_last else list(range(0, len(dataset)))
        if shuffle:
            np.random.shuffle(self.batch_indices)

    def __iter__(self):
        for indices in self.batch_indices:
            transposed = zip(*[self.dataset[idx] for idx in indices])
            batched = []
            for sample in transposed:
                if isinstance(sample[0], dgl.DGLGraph):
                    batched.append(dgl.batch(list(sample)))
                elif isinstance(sample[0], torch.Tensor):
                    batched.append(torch.stack(sample))
                elif isinstance(sample[0], MultipleTensors):
                    sample_ = MultipleTensors(
                        [pad_sequence([sample[i][j] for i in range(len(sample))]).permute(1, 0, 2) for j in range(len(sample[0]))])
                    batched.append(sample_)
                else:
                    raise NotImplementedError
            yield batched

    def __len__(self):
        return len(self.batch_indices)



