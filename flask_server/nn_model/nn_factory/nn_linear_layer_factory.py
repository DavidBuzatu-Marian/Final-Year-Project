import torch
import torch.nn as nn


class NNLinearLayerFactory:
    def get_layer(self, layer_type, parameters):
        options = {
            "Identity": self.__build_identity(parameters),
            "Linear": self.__build_linear(parameters),
            "Bilinear": self.__build_bilinear(parameters),
            "LazyLinear": self.__build_lazy_linear(parameters),
        }
        if layer_type in options:
            return options[layer_type]
        raise Exception("Layer type not in options")

    def __build_identity(parameters):
        return nn.Identity()

    def __build_linear(parameters):
        parameters_set = set("in_features", "out_features", "bias")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.Linear(**parameters)

    def __build_bilinear(parameters):
        parameters_set = set("in1_features", "in2_features", "out_features", "bias")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.Bilinear(**parameters)

    def __build_lazy_linear(parameters):
        parameters_set = set("out_features", "bias")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.LazyLinear(**parameters)
