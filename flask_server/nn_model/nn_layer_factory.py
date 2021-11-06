import torch
import torch.nn as nn

class NNLayerFactory():
    def get_layer(self, layer_type, parameters):
        match layer_type:
            case "Conv1d":
                return self.__build_conv1d(parameters)


    def __build_conv1d(parameters):
        conv1d_dict = set(
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "padding_mode",
            "dilation",
            "groups",
            "bias"
        )
        parameters = dict(filter(lambda key_value: not (key_value[0] in conv1d_dict), parameters.items()))
        return nn.Conv1d(**parameters)
