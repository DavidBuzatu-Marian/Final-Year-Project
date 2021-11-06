import sys

sys.path.insert(0, "../../../nn_model")

from nn_factory.nn_layer_factory import NNAbstractLayerFactory
import torch
import torch.nn as nn


class NNPoolingLayerFactory(NNAbstractLayerFactory):
    def get_layer(self, layer_type, parameters):
        options = {
            "MaxPool1d": self.__build_maxpool1d,
            "MaxPool2d": self.__build_maxpool2d,
            "MaxPool3d": self.__build_maxpool3d,
            "MaxUnpool1d": self.__build_maxunpool1d,
            "MaxUnpool2d": self.__build_maxunpool2d,
            "MaxUnpool3d": self.__build_maxunpool3d,
            "AvgPool1d": self.__build_avgpool1d,
            "AvgPool2d": self.__build_avgpool2d,
            "AvgPool3d": self.__build_avgpool3d,
            "FractionalMaxPool2d": self.__build_fractional_maxpool2d,
            "FractionalMaxPool3d": self.__build_fractional_maxpool3d,
            "LPPool1d": self.__build_lppool1d,
            "LPPool2d": self.__build_lppool2d,
            "AdaptiveMaxPool1d": self.__build_adaptive_maxpool1d,
            "AdaptiveMaxPool2d": self.__build_adaptive_maxpool2d,
            "AdaptiveMaxPool3d": self.__build_adaptive_maxpool3d,
            "AdaptiveAvgPool1d": self.__build_adaptive_avgpool1d,
            "AdaptiveAvgPool2d": self.__build_adaptive_avgpool2d,
            "AdaptiveAvgPool3d": self.__build_adaptive_avgpool3d,
        }
        if layer_type in options:
            return options[layer_type](parameters)
        raise Exception("Layer type not in options")

    def __build_maxpool1d(self, parameters):
        parameters_set = set(
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "return_indices",
            "ceil_mode",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.MaxPool1d(**parameters)

    def __build_maxpool2d(self, parameters):
        parameters_set = set(
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "return_indices",
            "ceil_mode",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.MaxPool2d(**parameters)

    def __build_maxpool3d(self, parameters):
        parameters_set = set(
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "return_indices",
            "ceil_mode",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.MaxPool3d(**parameters)

    def __build_maxunpool1d(self, parameters):
        parameters_set = set("kernel_size", "stride", "padding")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.MaxUnpool1d(**parameters)

    def __build_maxunpool2d(self, parameters):
        parameters_set = set("kernel_size", "stride", "padding")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.MaxUnpool2d(**parameters)

    def __build_maxunpool3d(self, parameters):
        parameters_set = set("kernel_size", "stride", "padding")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.MaxUnpool3d(**parameters)

    def __build_avgpool1d(self, parameters):
        parameters_set = set(
            "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AvgPool1d(**parameters)

    def __build_avgpool2d(self, parameters):
        parameters_set = set(
            "kernel_size",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
            "divisor_override",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AvgPool2d(**parameters)

    def __build_avgpool3d(self, parameters):
        parameters_set = set(
            "kernel_size",
            "stride",
            "padding",
            "ceil_mode",
            "count_include_pad",
            "divisor_override",
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AvgPool3d(**parameters)

    def __build_fractional_maxpool2d(self, parameters):
        parameters_set = set(
            "kernel_size", "output_size", "output_ratio", "return_indices"
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.FractionalMaxPool2d(**parameters)

    def __build_fractional_maxpool3d(self, parameters):
        parameters_set = set(
            "kernel_size", "output_size", "output_ratio", "return_indices"
        )
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.FractionalMaxPool3d(**parameters)

    def __build_lppool1d(self, parameters):
        parameters_set = set("kernel_size", "stride", "ceil_mode")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.LPPool1d(**parameters)

    def __build_lppool2d(self, parameters):
        parameters_set = set("kernel_size", "stride", "ceil_mode")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.LPPool2d(**parameters)

    def __build_adaptive_maxpool1d(self, parameters):
        parameters_set = set("output_size", "return_indices")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveMaxPool1d(**parameters)

    def __build_adaptive_maxpool2d(self, parameters):
        parameters_set = set("output_size", "return_indices")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveMaxPool2d(**parameters)

    def __build_adaptive_maxpool3d(self, parameters):
        parameters_set = set("output_size", "return_indices")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveMaxPool3d(**parameters)

    def __build_adaptive_avgpool1d(self, parameters):
        parameters_set = set("output_size")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveAvgPool1d(**parameters)

    def __build_adaptive_avgpool2d(self, parameters):
        parameters_set = set("output_size")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveAvgPool2d(**parameters)

    def __build_adaptive_avgpool3d(self, parameters):
        parameters_set = set("output_size")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveAvgPool3d(**parameters)
