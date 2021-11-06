import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, layer_types, parameters, activations_types):
        super(NNModel, self).__init__()  # needed for how torch works

        for layer, layer_params in zip(layer_types, parameters):
            self.layers.append(NNLayerFactory(layer, layer_params))
        for activation_type in activations_types:
            self.activations.append(NNActivationFactory(activation_type))

    def forward(self, input_data):
        for layer, activation in zip(self.layers, self.activations):
            input_data = activation(layer(input_data))
        return input_data
