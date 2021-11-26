import unittest
import torch
import torch.nn as nn

import sys


sys.path.insert(0, "../../nn_optimizer")
sys.path.insert(1, "../../nn_model")
from nn_optimizer_factory import NNOptimizerFactory
from nn_model import NNModel


class TestOptimizerFactory(unittest.TestCase):
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

    def test_sgd_creation(self):
        optimizer_factory = NNOptimizerFactory()
        sgd = optimizer_factory.get_optimizer(
            "SGD", {"params": self.model, "lr": 1e-3, "momentum": 0.9, "nesterov": True}
        )

        self.assertEqual(
            repr(sgd),
            repr(
                torch.optim.SGD(
                    self.model.parameters(), 1e-3, momentum=0.9, nesterov=True
                )
            ),
        )
