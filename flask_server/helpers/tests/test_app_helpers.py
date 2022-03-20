import unittest
import torch
import torch.nn as nn
import shutil
import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model_factory")
sys.path.insert(2, "../../nn_loss")
sys.path.insert(3, "../")
sys.path.insert(4, "../../nn_model_factory/model")

from app_helpers import *
from nn_model_factory.nn_model import NNModel


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
            read_model_from_path("./no_model.pth")

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
            repr(optimizer),
            repr(get_optimizer(request_json=request_json, model=self.model)),
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

    def test_get_hyperparameters_1(self):
        request_json = {"hyperparameters": {"epochs": 5, "num_workers": 2}}
        hyperparameters = get_hyperparameters(request_json=request_json)
        self.assertEqual(
            {"epochs": 5, "num_workers": 2, "batch_size": 1, "shuffle": True, "drop_last": False},
            hyperparameters,
        )

    def test_get_hyperparameters_2(self):
        request_json = {"hyperparameters": {"shuffle": True, "num_workers": 2}}
        hyperparameters = get_hyperparameters(request_json=request_json)
        self.assertEqual(
            {"epochs": 10, "num_workers": 2, "batch_size": 1, "shuffle": True, "drop_last": False},
            hyperparameters,
        )

    def test_get_hyperparameters_3(self):
        request_json = {
            "hyperparameters": {"batch_size": 10, "num_workers": 1, "epochs": 20}
        }
        hyperparameters = get_hyperparameters(request_json=request_json)
        self.assertEqual(
            {"epochs": 20, "num_workers": 1, "batch_size": 10, "shuffle": True, "drop_last": False},
            hyperparameters,
        )

    def test_get_instance_probability_of_failure(self):
        probability_of_failure = get_instance_probability_of_failure(
            config_path_env="INSTANCE_CONFIG_TEST_FILE_PATH")
        self.assertEqual(0.01, probability_of_failure)

    def test_get_instance_probability_of_failure_no_config(self):
        with self.assertRaises(Exception):
            get_instance_probability_of_failure(config_path_env="INVALID_ENV")

    def test_get_loss_type_training(self):
        request_json = {
            "loss_type": "training",
        }
        path = get_loss_type(request_json)
        self.assertEqual({"data_path": "TRAIN_DATA_PATH", "labels_path": "TRAIN_LABELS_PATH"},
                         path)

    def test_get_loss_type_test(self):
        request_json = {
            "loss_type": "test",
        }
        path = get_loss_type(request_json)
        self.assertEqual({"data_path": "TEST_DATA_PATH", "labels_path": "TEST_LABELS_PATH"},
                         path)

    def test_get_loss_type_validation(self):
        request_json = {
            "loss_type": "validation",
        }
        path = get_loss_type(request_json)
        self.assertEqual({
            "data_path": "VALIDATION_DATA_PATH",
            "labels_path": "VALIDATION_LABELS_PATH",
        },
            path)
