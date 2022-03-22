import sys


from helpers.nn_helpers import get_params_from_list
from nn_model_factory.nn_factory.nn_layer_factory import NNAbstractLayerFactory

import torch
import torch.nn as nn


class NNVisionLayerFactory(NNAbstractLayerFactory):
    def get_layer(self, layer_type, parameters):
        options = {
            "Upsample": self.__build_upsample,
        }
        if layer_type in options:
            return options[layer_type](parameters)
        raise Exception("Vision layer type not in options")

    def __build_upsample(self, parameters):
        parameters_set = {
            "size",
            "scale_factor",
            "mode",
            "align_corners",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Upsample(**parameters)
