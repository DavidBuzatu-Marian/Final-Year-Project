import torch
import torch.nn as nn


class NNPoolingLayerFactory:
    def get_layer(self, layer_type, parameters):
        options = {
            "MaxPool1d": self.__build_maxpool1d(parameters),
            "MaxPool2d": self.__build_maxpool2d(parameters),
            "MaxPool3d": self.__build_maxpool3d(parameters),
            "MaxUnpool1d": self.__build_maxunpool1d(parameters),
            "MaxUnpool2d": self.__build_maxunpool2d(parameters),
            "MaxUnpool3d": self.__build_maxunpool3d(parameters),
            "AvgPool1d": self.__build_avgpool1d(parameters),
            "AvgPool2d": self.__build_avgpool2d(parameters),
            "AvgPool3d": self.__build_avgpool3d(parameters),
            "FractionalMaxPool2d": self.__build_fractional_maxpool2d(parameters),
            "FractionalMaxPool3d": self.__build_fractional_maxpool3d(parameters),
            "LPPool1d": self.__build_lppool1d(parameters),
            "LPPool2d": self.__build_lppool2d(parameters),
            "AdaptiveMaxPool1d": self.__build_adaptive_maxpool1d(parameters),
            "AdaptiveMaxPool2d": self.__build_adaptive_maxpool2d(parameters),
            "AdaptiveMaxPool3d": self.__build_adaptive_maxpool3d(parameters),
            "AdaptiveAvgPool1d": self.__build_adaptive_avgpool1d(parameters),
            "AdaptiveAvgPool2d": self.__build_adaptive_avgpool2d(parameters),
            "AdaptiveAvgPool3d": self.__build_adaptive_avgpool3d(parameters),
        }
        if layer_type in options:
            return options[layer_type]
        raise Exception("Layer type not in options")

    def __build_maxpool1d(parameters):
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

    def __build_maxpool2d(parameters):
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

    def __build_maxpool3d(parameters):
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

    def __build_maxunpool1d(parameters):
        parameters_set = set("kernel_size", "stride", "padding")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.MaxUnpool1d(**parameters)

    def __build_maxunpool2d(parameters):
        parameters_set = set("kernel_size", "stride", "padding")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.MaxUnpool2d(**parameters)

    def __build_maxunpool3d(parameters):
        parameters_set = set("kernel_size", "stride", "padding")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.MaxUnpool3d(**parameters)

    def __build_avgpool1d(parameters):
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

    def __build_avgpool2d(parameters):
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

    def __build_avgpool3d(parameters):
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

    def __build_fractional_maxpool2d(parameters):
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

    def __build_fractional_maxpool3d(parameters):
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

    def __build_lppool1d(parameters):
        parameters_set = set("kernel_size", "stride", "ceil_mode")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.LPPool1d(**parameters)

    def __build_lppool2d(parameters):
        parameters_set = set("kernel_size", "stride", "ceil_mode")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.LPPool2d(**parameters)

    def __build_adaptive_maxpool1d(parameters):
        parameters_set = set("output_size", "return_indices")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveMaxPool1d(**parameters)

    def __build_adaptive_maxpool2d(parameters):
        parameters_set = set("output_size", "return_indices")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveMaxPool2d(**parameters)

    def __build_adaptive_maxpool3d(parameters):
        parameters_set = set("output_size", "return_indices")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveMaxPool3d(**parameters)

    def __build_adaptive_avgpool1d(parameters):
        parameters_set = set("output_size")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveAvgPool1d(**parameters)

    def __build_adaptive_avgpool2d(parameters):
        parameters_set = set("output_size")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveAvgPool2d(**parameters)

    def __build_adaptive_avgpool3d(parameters):
        parameters_set = set("output_size")
        parameters = dict(
            filter(
                lambda key_value: not (key_value[0] in parameters_set),
                parameters.items(),
            )
        )
        return nn.AdaptiveAvgPool3d(**parameters)
