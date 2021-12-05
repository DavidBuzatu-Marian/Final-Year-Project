import torch
import torch.nn as nn


from nn_factory.nn_convolution_layer_factory import NNConvolutionLayerFactory
from nn_factory.nn_pooling_layer_factory import NNPoolingLayerFactory
from nn_factory.nn_linear_layer_factory import NNLinearLayerFactory


class NNLayerFactory:
    # TODO: May want to encapsulate layer_type and rest in a wrapper class
    def get_layer(self, layer_type, subtype, parameters):
        options = {
            "Convolution": self.__build_convolution,
            "Pooling": self.__build_pooling,
            "Linear": self.__build_linear,
        }
        if layer_type in options:
            return options[layer_type](subtype, parameters)
        raise Exception("Layer type not in options")

    def __build_convolution(self, subtype, parameters):
        convolutional_layer_factory = NNConvolutionLayerFactory()
        return convolutional_layer_factory.get_layer(subtype, parameters)

    def __build_pooling(self, subtype, parameters):
        pooling_layer_factory = NNPoolingLayerFactory()
        return pooling_layer_factory.get_layer(subtype, parameters)

    def __build_linear(self, subtype, parameters):
        linear_layer_factory = NNLinearLayerFactory()
        return linear_layer_factory.get_layer(subtype, parameters)
