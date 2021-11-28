import unittest
import torch
import torch.nn as nn

import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model")

from nn_model.nn_layer_factory import NNLayerFactory


class TestLinearLayerFactory(unittest.TestCase):
    def test_identity_creation(self):
        layer_factory = NNLayerFactory()
        identity_layer = layer_factory.get_layer(
            layer_type="Linear",
            subtype="Identity",
            parameters={
                "in_channels": 16,  # is ignored
            },
        )
        self.assertEqual(repr(identity_layer), repr(nn.Identity()))

    def test_linear_creation(self):
        layer_factory = NNLayerFactory()
        linear_layer = layer_factory.get_layer(
            layer_type="Linear",
            subtype="Linear",
            parameters={"in_features": 20, "out_features": 30},
        )
        self.assertEqual(repr(linear_layer), repr(nn.Linear(20, 30)))

    def test_bilinear_creation(self):
        layer_factory = NNLayerFactory()
        bilinear_layer = layer_factory.get_layer(
            layer_type="Linear",
            subtype="Bilinear",
            parameters={"in1_features": 20, "in2_features": 30, "out_features": 40},
        )
        self.assertEqual(repr(bilinear_layer), repr(nn.Bilinear(20, 30, 40)))

    def test_lazy_creation(self):
        layer_factory = NNLayerFactory()
        lazy_layer = layer_factory.get_layer(
            layer_type="Linear",
            subtype="LazyLinear",
            parameters={"out_features": 40},
        )
        self.assertEqual(repr(lazy_layer), repr(nn.LazyLinear(40)))

    # Test with incorrect layer_type
    def test_linear_invalid_layer_type(self):
        layer_factory = NNLayerFactory()
        with self.assertRaises(Exception, msg="Layer type not in options"):
            layer_factory.get_layer(
                layer_type="Linar",
                subtype="Linear",
                parameters={"in_features": 20, "out_features": 30},
            )

    # Test with incorrect subtype
    def test_linear_invalid_subtype(self):
        layer_factory = NNLayerFactory()
        with self.assertRaises(Exception, msg="Linear type not in options"):
            layer_factory.get_layer(
                layer_type="Linear",
                subtype="Linar",
                parameters={"in_features": 20, "out_features": 30},
            )
