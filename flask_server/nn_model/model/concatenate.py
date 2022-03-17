from logging import error
import torch
import torch.nn as nn


class Concatenate(nn.Module):
    def __init__(self, previous_layer_index, dim):
        super().__init__()
        self.previous_layer_index = previous_layer_index
        self.dim = dim

    def get_previous_layer_index(self):
        return self.previous_layer_index

    def forward(self, previous_layer, current_layer):
        error(previous_layer.shape)
        error(current_layer.shape)
        return torch.cat([previous_layer, current_layer], dim=self.dim)
