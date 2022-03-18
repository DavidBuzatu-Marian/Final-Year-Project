import unittest
import torch
import torch.nn as nn

import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model")

from nn_model.nn_layer_factory import NNLayerFactory


class TestVisionLayerFactory(unittest.TestCase):
    def test_upsample_creation_without_params(self):
        layer_factory = NNLayerFactory()
        upsample_layer = layer_factory.get_layer(
            layer_type="Vision",
            subtype="Upsample",
            parameters={},
        )
        self.assertEqual(repr(upsample_layer), repr(nn.Upsample()))

    def test_upsample_creation_with_params(self):
        layer_factory = NNLayerFactory()
        upsample_layer = layer_factory.get_layer(
            layer_type="Vision",
            subtype="Upsample",
            parameters={
                "scale_factor": 2,
                "mode": 'bilinear',
                "align_corners": True
            },
        )
        self.assertEqual(repr(upsample_layer), repr(nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)))
