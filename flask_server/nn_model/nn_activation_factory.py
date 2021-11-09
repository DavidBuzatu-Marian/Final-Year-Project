import torch
import torch.nn as nn

# from nn_factory.nn_convolution_layer_factory import NNConvolutionLayerFactory
# from nn_factory.nn_pooling_layer_factory import NNPoolingLayerFactory
# from nn_factory.nn_linear_layer_factory import NNLinearLayerFactory
from nn_factory.nn_helpers import get_params_from_list


class NNActivationFactory:
    # TODO: May want to encapsulate activation_type and parameters in a class
    def get_activation(self, activation_type, parameters):
        options = {
            "ELU": self.__build_elu,
            "Hardshrink": self.__build_hardshrink,
            "Hardsigmoid": self.__build_hardsigmoid,
            "Hardtanh": self.__build_hardtanh,
            "Hardswish": self.__build_hardswish,
            "LeakyReLU": self.__build_leakyrelu,
            "LogSigmoid": self.__build_logsigmoid,
            "ReLU": self.__build_relu,
            "CELU": self.__build_celu,
            "SELU": self.__build_selu,
            "Sigmoid": self.__build_sigmoid,
            "SiLU": self.__build_silu,
            "Tanh": self.__build_tanh,
            "Softmin": self.__build_softmin,
            "Softmax": self.__build_softmax,
        }
        if activation_type in options:
            return options[activation_type](parameters)
        raise Exception("Layer type not in options")

    def __build_elu(self, parameters):
        parameters_set = {"alpha", "inplace"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.ELU(**parameters)

    def __build_hardshrink(self, parameters):
        parameters_set = {"lambd"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Hardshrink(**parameters)

    def __build_hardsigmoid(self, parameters):
        parameters_set = {"inplace"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Hardsigmoid(**parameters)

    def __build_hardtanh(self, parameters):
        parameters_set = {"min_val", "max_val", "inplace"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Hardtanh(**parameters)

    def __build_hardswish(self, parameters):
        parameters_set = {"inplace"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Hardswish(**parameters)

    def __build_leakyrelu(self, parameters):
        parameters_set = {"negative_slope", "inplace"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.LeakyRELU(**parameters)

    def __build_logsigmoid(self, parameters):
        parameters_set = {}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.LogSigmoid(**parameters)

    def __build_relu(self, parameters):
        parameters_set = {"inplace"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.ReLU(**parameters)

    def __build_selu(self, parameters):
        parameters_set = {"inplace"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.SELU(**parameters)

    def __build_celu(self, parameters):
        parameters_set = {"alpha", "inplace"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.CELU(**parameters)

    def __build_gelu(self, parameters):
        parameters_set = {}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.GELU(**parameters)

    def __build_sigmoid(self, parameters):
        parameters_set = {}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Sigmoid(**parameters)

    def __build_tanh(self, parameters):
        parameters_set = {}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Tanh(**parameters)

    def __build_silu(self, parameters):
        parameters_set = {}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.SiLU(**parameters)

    def __build_softmax(self, parameters):
        parameters_set = {"dim"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Softmax(**parameters)

    def __build_softmin(self, parameters):
        parameters_set = {"dim"}
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Softmin(**parameters)
