import torch
import torch.nn as nn


class Concatenate(nn.Module):
    def __init__(self, previous_layer, current_layer, dim):
        self.previous_layer = previous_layer
        self.current_layer = current_layer
        self.dim = dim

    def forward(self):
        return torch.cat([self.previous_layer, self.current_layer], dim=self.dim)
