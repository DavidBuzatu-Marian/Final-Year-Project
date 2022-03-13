import unittest
import torch
import torch.nn as nn

import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model")
sys.path.insert(2, "../")

from nn_model.nn_activation_factory import NNActivationFactory


class TestActivationLayerFactory(unittest.TestCase):
    def test_elu_creation(self):
        activation_factory = NNActivationFactory()
        elu = activation_factory.get_activation(
            activation_type="ELU", parameters={"alpha": 1.5}
        )

        self.assertEqual(repr(elu), repr(nn.ELU(alpha=1.5)))

    def test_hardshrink_creation(self):
        activation_factory = NNActivationFactory()
        hardshrink = activation_factory.get_activation(
            activation_type="Hardshrink", parameters={"lambd": 1.5}
        )

        self.assertEqual(repr(hardshrink), repr(nn.Hardshrink(lambd=1.5)))

    def test_hardsigmoid_creation(self):
        activation_factory = NNActivationFactory()
        hardsigmoid = activation_factory.get_activation(
            activation_type="Hardsigmoid", parameters={}
        )

        self.assertEqual(repr(hardsigmoid), repr(nn.Hardsigmoid()))

    def test_hardtanh_creation(self):
        activation_factory = NNActivationFactory()
        hardtanh = activation_factory.get_activation(
            activation_type="Hardtanh", parameters={"min_val": -1, "max_val": 1}
        )

        self.assertEqual(repr(hardtanh), repr(nn.Hardtanh(-1, 1)))

    def test_leaky_relu_creation(self):
        activation_factory = NNActivationFactory()
        leaky_relu = activation_factory.get_activation(
            activation_type="LeakyReLU", parameters={"negative_slope": -1}
        )

        self.assertEqual(repr(leaky_relu), repr(nn.LeakyReLU(-1)))

    def test_log_sigmoid_creation(self):
        activation_factory = NNActivationFactory()
        log_sigmoid = activation_factory.get_activation(
            activation_type="LogSigmoid", parameters={}
        )

        self.assertEqual(repr(log_sigmoid), repr(nn.LogSigmoid()))

    def test_relu_creation(self):
        activation_factory = NNActivationFactory()
        relu = activation_factory.get_activation(activation_type="ReLU", parameters={})

        self.assertEqual(repr(relu), repr(nn.ReLU()))

    def test_celu_creation(self):
        activation_factory = NNActivationFactory()
        celu = activation_factory.get_activation(
            activation_type="CELU", parameters={"alpha": 1.5}
        )

        self.assertEqual(repr(celu), repr(nn.CELU(1.5)))

    def test_sigmoid_creation(self):
        activation_factory = NNActivationFactory()
        sigmoid = activation_factory.get_activation(
            activation_type="Sigmoid", parameters={}
        )

        self.assertEqual(repr(sigmoid), repr(nn.Sigmoid()))

    def test_selu_creation(self):
        activation_factory = NNActivationFactory()
        selu = activation_factory.get_activation(activation_type="SELU", parameters={})

        self.assertEqual(repr(selu), repr(nn.SELU()))

    def test_tanh_creation(self):
        activation_factory = NNActivationFactory()
        tanh = activation_factory.get_activation(activation_type="Tanh", parameters={})

        self.assertEqual(repr(tanh), repr(nn.Tanh()))

    def test_softmin_creation(self):
        activation_factory = NNActivationFactory()
        softmin = activation_factory.get_activation(
            activation_type="Softmin", parameters={"dim": 1}
        )

        self.assertEqual(repr(softmin), repr(nn.Softmin(dim=1)))

    def test_softmax_creation(self):
        activation_factory = NNActivationFactory()
        softmax = activation_factory.get_activation(
            activation_type="Softmax", parameters={"dim": 1}
        )

        self.assertEqual(repr(softmax), repr(nn.Softmax(dim=1)))
