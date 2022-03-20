import sys


from helpers.nn_helpers import get_params_from_list
from nn_model_factory.nn_factory.nn_layer_factory import NNAbstractLayerFactory

import torch
import torch.nn as nn


class NNLinearLayerFactory(NNAbstractLayerFactory):
    def get_layer(self, layer_type, parameters):
        options = {
            "Identity": self.__build_identity,
            "Linear": self.__build_linear,
            "Bilinear": self.__build_bilinear,
            "LazyLinear": self.__build_lazy_linear,
        }
        if layer_type in options:
            return options[layer_type](parameters)
        raise Exception("Linear layer type not in options")

    def __build_identity(self, parameters):
        return nn.Identity()

    def __build_linear(self, parameters):
        parameters_set = {"in_features", "out_features", "bias"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Linear(**parameters)

    def __build_bilinear(self, parameters):
        parameters_set = {"in1_features", "in2_features", "out_features", "bias"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Bilinear(**parameters)

    def __build_lazy_linear(self, parameters):
        parameters_set = {"out_features", "bias"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.LazyLinear(**parameters)
