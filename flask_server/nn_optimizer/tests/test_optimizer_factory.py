import unittest
import torch
import torch.nn as nn

import sys


sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model")
sys.path.insert(2, "../../nn_optimizer")

from nn_optimizer.nn_optimizer_factory import NNOptimizerFactory
from nn_model.nn_model import NNModel


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

    def test_rmsprop_creation(self):
        optimizer_factory = NNOptimizerFactory()
        rmsprop = optimizer_factory.get_optimizer(
            "RMSprop",
            {"params": self.model, "lr": 1e-3, "momentum": 0.9, "weight_decay": 1e-8},
        )

        self.assertEqual(
            repr(rmsprop),
            repr(
                torch.optim.RMSprop(
                    self.model.parameters(), 1e-3, momentum=0.9, weight_decay=1e-8
                )
            ),
        )

    def test_asgd_creation(self):
        optimizer_factory = NNOptimizerFactory()
        asgd = optimizer_factory.get_optimizer(
            "ASGD",
            {"params": self.model, "lr": 1e-3, "alpha": 0.9, "weight_decay": 1e-8},
        )

        self.assertEqual(
            repr(asgd),
            repr(
                torch.optim.ASGD(
                    self.model.parameters(), 1e-3, alpha=0.9, weight_decay=1e-8
                )
            ),
        )

    def test_adamax_creation(self):
        optimizer_factory = NNOptimizerFactory()
        adamax = optimizer_factory.get_optimizer(
            "Adamax",
            {
                "params": self.model,
                "lr": 1e-3,
                "betas": (0.9, 0.998),
                "weight_decay": 1e-8,
            },
        )

        self.assertEqual(
            repr(adamax),
            repr(
                torch.optim.Adamax(
                    self.model.parameters(), 1e-3, betas=(0.9, 0.998), weight_decay=1e-8
                )
            ),
        )

    def test_adam_creation(self):
        optimizer_factory = NNOptimizerFactory()
        adam = optimizer_factory.get_optimizer(
            "Adam",
            {
                "params": self.model,
                "lr": 1e-3,
                "betas": (0.9, 0.998),
                "weight_decay": 1e-8,
            },
        )

        self.assertEqual(
            repr(adam),
            repr(
                torch.optim.Adam(
                    self.model.parameters(), 1e-3, betas=(0.9, 0.998), weight_decay=1e-8
                )
            ),
        )

    def test_adagrad_creation(self):
        optimizer_factory = NNOptimizerFactory()
        adagrad = optimizer_factory.get_optimizer(
            "Adagrad",
            {"params": self.model, "lr": 1e-3, "lr_decay": 2e-3, "weight_decay": 1e-8},
        )

        self.assertEqual(
            repr(adagrad),
            repr(
                torch.optim.Adagrad(
                    self.model.parameters(), 1e-3, lr_decay=2e-3, weight_decay=1e-8
                )
            ),
        )
