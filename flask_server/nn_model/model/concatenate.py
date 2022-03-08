import torch
import torch.nn as nn


class Concatenate(nn.Module):
    def __init__(self, previous_layer, dim):
        self.previous_layer = previous_layer
        self.dim = dim

    def forward(self, current_layer):
        return torch.cat([self.previous_layer, current_layer], dim=self.dim)
