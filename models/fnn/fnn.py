import torch
import torch.nn as nn
import numpy as np

class FNN(nn.Module):
    def __init__(self, in_shape, hidden_features, hidden_layers, out_shape:tuple) -> None:
        super().__init__()
        self.out_shape = out_shape
        in_features = np.prod(in_shape)
        out_features = np.prod(out_shape)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.in_layer = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.out_layer = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.layers = nn.ModuleList([nn.Linear(in_features=hidden_features, out_features=hidden_features) for i in range(hidden_layers)])
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.in_layer(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        x = self.out_layer(x)
        x = x.reshape((-1,) + self.out_shape)
        return x

