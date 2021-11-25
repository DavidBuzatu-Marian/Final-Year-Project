import unittest
import torch
import torch.nn as nn
from torchsummary import summary

import sys

sys.path.insert(0, "../../nn_model")
from nn_model import NNModel


class TestModel(unittest.TestCase):
    def test_single_layer_convolution(self):
        model = NNModel(
            {
                "network": [
                    {
                        "layer": {
                            "layer_type": "Convolution",
                            "subtype": "Conv2d",
                            "parameters": {
                                "in_channels": 16,
                                "out_channels": 33,
                                "kernel_size": 3,
                                "stride": 2,
                            },
                        }
                    }
                ]
            }
        )

        pytorch_model = nn.Sequential(nn.Conv2d(16, 33, 3, stride=2))
        self.assertEqual(len(model._model), len(pytorch_model))
        for idx in range(len(model._model)):
            self.assertEqual(repr(model._model[idx]), repr(pytorch_model[idx]))

    def test_single_convolution_with_activation_convolution(self):
        model = NNModel(
            {
                "network": [
                    {
                        "layer": {
                            "layer_type": "Convolution",
                            "subtype": "Conv2d",
                            "parameters": {
                                "in_channels": 16,
                                "out_channels": 33,
                                "kernel_size": 3,
                                "stride": 2,
                            },
                        }
                    },
                    {
                        "activation": {
                            "activation_type": "ReLU",
                            "parameters": {
                                "random": 16,
                            },
                        }
                    },
                ]
            }
        )

        pytorch_model = nn.Sequential(nn.Conv2d(16, 33, 3, stride=2), nn.ReLU())
        self.assertEqual(len(model._model), len(pytorch_model))
        for idx in range(len(model._model)):
            self.assertEqual(repr(model._model[idx]), repr(pytorch_model[idx]))

    def test_double_convolution_with_activation_convolution(self):
        model = NNModel(
            {
                "network": [
                    {
                        "layer": {
                            "layer_type": "Convolution",
                            "subtype": "Conv2d",
                            "parameters": {
                                "in_channels": 16,
                                "out_channels": 33,
                                "kernel_size": 3,
                                "stride": 2,
                            },
                        }
                    },
                    {
                        "activation": {
                            "activation_type": "ReLU",
                            "parameters": {
                                "random": 16,
                            },
                        }
                    },
                    {
                        "layer": {
                            "layer_type": "Convolution",
                            "subtype": "Conv2d",
                            "parameters": {
                                "in_channels": 16,
                                "out_channels": 33,
                                "kernel_size": 3,
                                "stride": 2,
                            },
                        }
                    },
                    {
                        "activation": {
                            "activation_type": "ReLU",
                            "parameters": {
                                "random": 16,
                            },
                        }
                    },
                ]
            }
        )

        pytorch_model = nn.Sequential(
            nn.Conv2d(16, 33, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 33, 3, stride=2),
            nn.ReLU(),
        )
        self.assertEqual(len(model._model), len(pytorch_model))
        for idx in range(len(model._model)):
            self.assertEqual(repr(model._model[idx]), repr(pytorch_model[idx]))
