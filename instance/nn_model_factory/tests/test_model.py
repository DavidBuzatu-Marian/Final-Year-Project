import unittest
import torch
import torch.nn as nn

import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model_factory")
from nn_model_factory.nn_model import NNModel


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
        self.assertEqual(len(model.nn_layers), len(pytorch_model))
        for idx in range(len(model.nn_layers)):
            self.assertEqual(repr(model.nn_layers[idx]), repr(pytorch_model[idx]))

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
        self.assertEqual(len(model.nn_layers), len(pytorch_model))
        for idx in range(len(model.nn_layers)):
            self.assertEqual(repr(model.nn_layers[idx]), repr(pytorch_model[idx]))

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
        self.assertEqual(len(model.nn_layers), len(pytorch_model))
        for idx in range(len(model.nn_layers)):
            self.assertEqual(repr(model.nn_layers[idx]), repr(pytorch_model[idx]))

    def test_double_convolution_with_activation_and_concatenation(self):
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
                    {
                        "layer": {
                            "layer_type": "Vision",
                            "subtype": "Upsample",
                            "parameters": {
                                "scale_factor": 2,
                                "mode": 'bilinear',
                                "align_corners": True
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
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.assertEqual(len(model.nn_layers), len(pytorch_model))
        for idx in range(len(model.nn_layers)):
            self.assertEqual(repr(model.nn_layers[idx]), repr(pytorch_model[idx]))
