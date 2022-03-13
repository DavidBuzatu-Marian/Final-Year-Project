from model.concatenate import Concatenate
from nn_activation_factory import NNActivationFactory
from nn_layer_factory import NNLayerFactory
import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, architecture_components):
        super(NNModel, self).__init__()  # needed for how torch works
        layer_factory = NNLayerFactory()
        activation_factory = NNActivationFactory()
        self._model = nn.ModuleList()

        for layer_dict in architecture_components["network"]:
            for component_type, component_details in layer_dict.items():
                if component_type == "layer":
                    self._model.append(
                        layer_factory.get_layer(
                            component_details["layer_type"],
                            component_details["subtype"],
                            component_details["parameters"],
                        )
                    )
                elif component_type == "activation":
                    self._model.append(
                        activation_factory.get_activation(
                            component_details["activation_type"],
                            component_details["parameters"],
                        )
                    )
                elif component_type == "concatenate":
                    self._model.append(
                        Concatenate(
                            component_details["previous_layer_index"],
                            component_details["dim"]))
                elif component_type == "optimizer":
                    pass
                else:
                    raise Exception("Invalid component type")

    def forward(self, input_data):
        computed_layers = list()
        prediction = input_data
        for component in self._model:
            if isinstance(component, Concatenate):
                prediction = component(
                    computed_layers[component.get_previous_layer_index()],
                    prediction)
            else:
                prediction = component(prediction)
            computed_layers.append(prediction)
        return prediction
