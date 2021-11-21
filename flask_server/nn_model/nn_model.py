from flask_server.nn_model.nn_activation_factory import NNActivationFactory
from flask_server.nn_model.nn_layer_factory import NNLayerFactory
import torch
import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, architecture_components):
        super(NNModel, self).__init__()  # needed for how torch works
        layer_factory = NNLayerFactory()
        activation_factory = NNActivationFactory()
        for component_type, component_details in architecture_components.items():
            if component_type == "layer":
                self.model.append(
                    layer_factory.get_layer(
                        component_details["layer_type"],
                        component_details["subtype"],
                        component_details["parameters"],
                    )
                )
            elif component_type == "activation":
                self.model.append(
                    activation_factory.get_activation(
                        component_details["activation_type"],
                        component_details["parameters"],
                    )
                )
            elif component_type == "concatenate":
                pass
            elif component_type == "optimizer":
                pass
            else:
                raise Exception("Invalid component type")

    def forward(self, input_data):
        prediction = input_data
        for component in self.model:
            prediction = component(prediction)
        return prediction
