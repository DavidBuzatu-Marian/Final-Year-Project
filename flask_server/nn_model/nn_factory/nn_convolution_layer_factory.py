from flask_server.nn_model.nn_factory.nn_layer_factory import NNAbstractLayerFactory
import torch
import torch.nn as nn


class NNConvolutionLayerFactory(NNAbstractLayerFactory):
    def get_layer(self, layer_type, parameters):
        options = {
            "Conv1d": self.__build_conv1d(parameters),
            "Conv2d": self.__build_conv2d(parameters),
            "Conv3d": self.__build_conv3d(parameters),
            "ConvTranspose1d": self.__build_conv_transpose1d(parameters),
            "ConvTranspose2d": self.__build_conv_transpose2d(parameters),
            "ConvTranspose3d": self.__build_conv_transpose3d(parameters),
            "Fold": self.__build_fold(parameters),
            "Unfold": self.__build_unfold(parameters),
        }
        if layer_type in options:
            return options[layer_type]
        raise Exception("Layer type not in options")

    def __build_conv1d(parameters):
        conv1d_set = set(
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "padding_mode",
            "dilation",
            "groups",
            "bias",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in conv1d_set), parameters.items()
            )
        )
        return nn.Conv1d(**parameters)

    def __build_conv2d(parameters):
        conv2d_set = set(
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "padding_mode",
            "dilation",
            "groups",
            "bias",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in conv2d_set), parameters.items()
            )
        )
        return nn.Conv2d(**parameters)

    def __build_conv3d(parameters):
        conv3d_set = set(
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "padding_mode",
            "dilation",
            "groups",
            "bias",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in conv3d_set), parameters.items()
            )
        )
        return nn.Conv3d(**parameters)

    def __build_conv_transpose1d(parameters):
        conv_transpose1d_set = set(
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "groups",
            "bias",
            "dilation",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in conv_transpose1d_set),
                parameters.items(),
            )
        )
        return nn.ConvTranspose1d(**parameters)

    def __build_conv_transpose2d(parameters):
        conv_transpose2d_set = set(
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "groups",
            "bias",
            "dilation",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in conv_transpose2d_set),
                parameters.items(),
            )
        )
        return nn.ConvTranspose2d(**parameters)

    def __build_conv_transpose3d(parameters):
        conv_transpose3d_set = set(
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "output_padding",
            "groups",
            "bias",
            "dilation",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in conv_transpose3d_set),
                parameters.items(),
            )
        )
        return nn.ConvTranspose3d(**parameters)

    def __build_fold(parameters):
        fold_set = set("output_size", "kernel_size", "stride", "padding", "dilation")

        parameters = dict(
            filter(lambda key_value: not (key_value[0] in fold_set), parameters.items())
        )
        return nn.Fold(**parameters)

    def __build_unfold(parameters):
        unfold_set = set("kernel_size", "stride", "padding", "dilation")

        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in unfold_set), parameters.items()
            )
        )
        return nn.Unfold(**parameters)
