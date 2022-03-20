import unittest
import torch
import torch.nn as nn
import shutil
import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model_factory")
sys.path.insert(2, "../../nn_model_factory/model")
sys.path.insert(3, "../")

from dotenv import load_dotenv
from data_helpers import *

load_dotenv()


class TestDataHelpers(unittest.TestCase):
    def test_dataloader(self):
        dl = get_dataloader(
            data_path="../../train_data",
            labels_path="../../train_labels",
            hyperparameters={"num_workers": 1, "batch_size": 1,
                             "shuffle": True, "drop_last": False},
        )
        for data, label in dl:
            self.assertEqual(data.shape, (1, 96, 96))
            self.assertEqual(label.shape, (1, 96, 96))

    def test_reshape(self):
        dl = get_dataloader(
            data_path="../../train_data",
            labels_path="../../train_labels",
            hyperparameters={"num_workers": 1, "batch_size": 1,
                             "shuffle": True, "drop_last": False},
        )
        for data, label in dl:
            self.assertEqual(data.shape, (1, 96, 96))
            self.assertEqual(label.shape, (1, 96, 96))

            self.assertEqual(
                reshape_data(data, {"reshape": "384, 24"}).shape, (384, 24)
            )
            self.assertEqual(reshape_data(label, {}).shape, (1, 96, 96))

    def test_reshape_no_reshape_hyperparameter(self):
        dl = get_dataloader(
            data_path="../../train_data",
            labels_path="../../train_labels",
            hyperparameters={"num_workers": 1, "batch_size": 1,
                             "shuffle": True, "drop_last": False},
        )
        for data, label in dl:
            self.assertEqual(data.shape, (1, 96, 96))
            self.assertEqual(label.shape, (1, 96, 96))

            self.assertEqual(reshape_data(data, {}).shape, (1, 96, 96))
            self.assertEqual(reshape_data(label, {}).shape, (1, 96, 96))

    def test_normalize(self):
        dl = get_dataloader(
            data_path="./mock_files/train_data",
            labels_path="./mock_files/train_labels",
            hyperparameters={"num_workers": 1, "batch_size": 1,
                             "shuffle": True, "drop_last": False},
        )
        mean, std = compute_mean_and_std(dl)
        for data, label in dl:
            self.assertEqual(data.shape, (1, 96, 96))
            self.assertEqual(label.shape, (1, 96, 96))
            norm_data = normalize_data(data, mean, std)
            norm_label = normalize_data(label, mean, std)
            self.assertLessEqual(
                torch.max(norm_data), 1.0
            )
            self.assertGreaterEqual(
                torch.min(norm_data), 0.0
            )
            self.assertLessEqual(
                torch.max(norm_label), 1.0
            )
            self.assertGreaterEqual(
                torch.min(norm_label), 0.0
            )

    def test_normalize_no_normalize_hyperparameter(self):
        dl = get_dataloader(
            data_path="../../train_data",
            labels_path="../../train_labels",
            hyperparameters={"num_workers": 1, "batch_size": 1,
                             "shuffle": True, "drop_last": False},
        )
        for data, label in dl:
            self.assertEqual(data.shape, (1, 96, 96))
            self.assertEqual(label.shape, (1, 96, 96))
            if not normalize({}):
                self.assertLessEqual(torch.max(data), 255.0)
                self.assertLessEqual(torch.max(label), 255.0)
