import torch
import torch.nn as nn
from nn_factory.nn_convolution_layer_factory import NNConvolutionLayerFactory


class NNLayerFactory:
    # TODO: May want to encapsulate layer_type and rest in a wrapper class
    def get_layer(self, layer_type, subtype, parameters):
        options = {
            "Convolution": self.__build_convolution(subtype, parameters),
        }
        if layer_type in options:
            return options[layer_type]
        raise Exception("Layer type not in options")

    def __build_convolution(subtype, parameters):
        return NNConvolutionLayerFactory.get_layer(subtype, parameters)
