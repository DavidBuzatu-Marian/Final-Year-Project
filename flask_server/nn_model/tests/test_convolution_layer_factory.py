import unittest
import torch
import torch.nn as nn

import sys

sys.path.insert(0, "../../nn_model")
from nn_layer_factory import NNLayerFactory


class TestConvolutionLayerFactory(unittest.TestCase):
    def test_conv1d_creation(self):
        layer_factory = NNLayerFactory()
        conv1d_layer = layer_factory.get_layer(
            layer_type="Convolution",
            subtype="Conv1d",
            parameters={
                "in_channels": 16,
                "out_channels": 33,
                "kernel_size": 3,
                "stride": 2,
            },
        )
        self.assertEqual(conv1d_layer, nn.Conv1d(16, 33, 3, stride=2))


if __name__ == "__main__":
    unittest.main()
