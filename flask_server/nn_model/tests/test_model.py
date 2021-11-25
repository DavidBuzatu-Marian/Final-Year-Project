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
        )

        pytorch_model = nn.Sequential(nn.Conv2d(16, 33, 3, stride=2))
        model.eval()

        for idx in range(len(model._model)):
            self.assertEqual(repr(model._model[idx]), repr(pytorch_model[idx]))

    def init_weights(self, model):
        for m in model.modules():
            torch.nn.init.xavier_uniform(m.weight)
