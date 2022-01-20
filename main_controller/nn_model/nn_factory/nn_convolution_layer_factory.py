import sys


from helpers.nn_helpers import get_params_from_list
from nn_factory.nn_layer_factory import NNAbstractLayerFactory
import torch
import torch.nn as nn


class NNConvolutionLayerFactory(NNAbstractLayerFactory):
    def get_layer(self, layer_type, parameters):
        options = {
            "Conv1d": self.__build_conv1d,
            "Conv2d": self.__build_conv2d,
            "Conv3d": self.__build_conv3d,
            "ConvTranspose1d": self.__build_conv_transpose1d,
            "ConvTranspose2d": self.__build_conv_transpose2d,
            "ConvTranspose3d": self.__build_conv_transpose3d,
            "Fold": self.__build_fold,
            "Unfold": self.__build_unfold,
        }
        if layer_type in options:
            return options[layer_type](parameters)
        raise Exception("Convolutional layer type not in options")

    def __build_conv1d(self, parameters):
        parameters_set = {
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "padding_mode",
            "dilation",
            "groups",
            "bias",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Conv1d(**parameters)

    def __build_conv2d(self, parameters):
        parameters_set = {
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "padding_mode",
            "dilation",
            "groups",
            "bias",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Conv2d(**parameters)

    def __build_conv3d(self, parameters):
        parameters_set = {
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "padding_mode",
            "dilation",
            "groups",
            "bias",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Conv3d(**parameters)

    def __build_conv_transpose1d(self, parameters):
        parameters_set = {
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "groups",
            "bias",
            "dilation",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.ConvTranspose1d(**parameters)

    def __build_conv_transpose2d(self, parameters):
        parameters_set = {
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "groups",
            "bias",
            "dilation",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.ConvTranspose2d(**parameters)

    def __build_conv_transpose3d(self, parameters):
        parameters_set = {
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "groups",
            "bias",
            "dilation",
        }
        parameters = get_params_from_list(parameters, parameters_set)
        return nn.ConvTranspose3d(**parameters)

    def __build_fold(self, parameters):
        parameters_set = {"output_size", "kernel_size", "stride", "padding", "dilation"}

        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Fold(**parameters)

    def __build_unfold(self, parameters):
        parameters_set = {"kernel_size", "stride", "padding", "dilation"}

        parameters = get_params_from_list(parameters, parameters_set)
        return nn.Unfold(**parameters)
