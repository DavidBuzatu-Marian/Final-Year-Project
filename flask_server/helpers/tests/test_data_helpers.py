import unittest
import torch
import torch.nn as nn
import shutil
import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model")
sys.path.insert(2, "../../nn_loss")
sys.path.insert(3, "../")
from dotenv import load_dotenv
from data_helpers import *

load_dotenv()


class TestDataHelpers(unittest.TestCase):
    def test_dataloader(self):
        dl = get_dataloader(
            data_path="../../train_data",
            labels_path="../../train_labels",
            hyperparameters={"num_workers": 1, "batch_size": 1, "shuffle": True},
        )
        for data, label in dl:
            self.assertEqual(data.shape, (1, 96, 96))
            self.assertEqual(label.shape, (1, 96, 96))

    def test_reshape(self):
        dl = get_dataloader(
            data_path="../../train_data",
            labels_path="../../train_labels",
            hyperparameters={"num_workers": 1, "batch_size": 1, "shuffle": True},
        )
        for data, label in dl:
            self.assertEqual(data.shape, (1, 96, 96))
            self.assertEqual(label.shape, (1, 96, 96))

            self.assertEqual(
                reshape_data(data, {"reshape": (384, 24)}).shape, (384, 24)
            )
            self.assertEqual(reshape_data(label, {}).shape, (1, 96, 96))

    def test_reshape_no_reshape_hyperparameter(self):
        dl = get_dataloader(
            data_path="../../train_data",
            labels_path="../../train_labels",
            hyperparameters={"num_workers": 1, "batch_size": 1, "shuffle": True},
        )
        for data, label in dl:
            self.assertEqual(data.shape, (1, 96, 96))
            self.assertEqual(label.shape, (1, 96, 96))

            self.assertEqual(reshape_data(data, {}).shape, (1, 96, 96))
            self.assertEqual(reshape_data(label, {}).shape, (1, 96, 96))

    def test_normalize(self):
        dl = get_dataloader(
            data_path="../../train_data",
            labels_path="../../train_labels",
            hyperparameters={"num_workers": 1, "batch_size": 1, "shuffle": True},
        )
        for data, label in dl:
            self.assertEqual(data.shape, (1, 96, 96))
            self.assertEqual(label.shape, (1, 96, 96))

            self.assertLessEqual(
                torch.max(normalize_data(data, {"normalizer": 255.0})), 1.0
            )
            self.assertLessEqual(
                torch.max(normalize_data(label, {"normalizer": 255.0})), 1.0
            )

    def test_normalize_no_normalize_hyperparameter(self):
        dl = get_dataloader(
            data_path="../../train_data",
            labels_path="../../train_labels",
            hyperparameters={"num_workers": 1, "batch_size": 1, "shuffle": True},
        )
        for data, label in dl:
            self.assertEqual(data.shape, (1, 96, 96))
            self.assertEqual(label.shape, (1, 96, 96))

            self.assertLessEqual(torch.max(normalize_data(data, {})), 255.0)
            self.assertLessEqual(torch.max(normalize_data(label, {})), 255.0)
