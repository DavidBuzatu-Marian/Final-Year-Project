import unittest
import torch
import torch.nn as nn
import shutil
import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model")
sys.path.insert(2, "../../nn_loss")
sys.path.insert(3, "../")

from app_helpers import *
from nn_model.nn_model import NNModel


class TestAppHelpers(unittest.TestCase):

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

    def test_read_model_with_no_model(self):
        with self.assertRaises(Exception):
            read_model_from_path("/no_model.pth")

    # def test_save_file_with_existing_dir(self):
    #     file = open("./__init__.py", "r")
    #     save_file(file, "./test.py")
    #     file.close()
    #     os.remove("./test.py")

    # def test_save_file_without_dir(self):
    #     file = open("./__init__.py", "r")
    #     file.filename = "__init__.py"
    #     save_file(file, "/test/test.py")
    #     file.close()
    #     shutil.rmtree("/test/test.py")

    def test_get_loss(self):
        request_json = {"loss": {"loss_type": "CrossEntropyLoss", "parameters": {}}}
        loss = torch.nn.CrossEntropyLoss()
        self.assertEqual(repr(loss), repr(get_loss(request_json=request_json)))

    def test_get_loss_exception(self):
        request_json = {"no_loss": {"loss_type": "CrossEntropyLoss", "parameters": {}}}
        with self.assertRaises(Exception):
            get_loss(request_json=request_json)

    def test_get_optimizer(self):
        request_json = {
            "optimizer": {
                "optimizer_type": "RMSprop",
                "parameters": {"params": self.model},
            }
        }
        optimizer = torch.optim.RMSprop(self.model.parameters())
        self.assertEqual(
            repr(optimizer), repr(get_optimizer(request_json=request_json))
        )

    def test_get_optimizer_exception(self):
        request_json = {
            "no_optimizer": {
                "optimizer_type": "RMSprop",
                "parameters": {"params": self.model},
            }
        }
        with self.assertRaises(Exception):
            get_optimizer(request_json=request_json)
