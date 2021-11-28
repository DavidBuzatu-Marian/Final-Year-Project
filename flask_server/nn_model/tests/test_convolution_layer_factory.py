import unittest
import torch
import torch.nn as nn

import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model")
from nn_model.nn_layer_factory import NNLayerFactory


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
        self.assertEqual(repr(conv1d_layer), repr(nn.Conv1d(16, 33, 3, stride=2)))

    def test_conv2d_creation(self):
        layer_factory = NNLayerFactory()
        conv2d_layer = layer_factory.get_layer(
            layer_type="Convolution",
            subtype="Conv2d",
            parameters={
                "in_channels": 16,
                "out_channels": 33,
                "kernel_size": (3, 5),
                "stride": (2, 1),
                "padding": (4, 2),
                "dilation": (3, 1),
            },
        )
        self.assertEqual(
            repr(conv2d_layer),
            repr(
                nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
            ),
        )

    def test_conv3d_creation(self):
        layer_factory = NNLayerFactory()
        conv3d_layer = layer_factory.get_layer(
            layer_type="Convolution",
            subtype="Conv3d",
            parameters={
                "in_channels": 16,
                "out_channels": 33,
                "kernel_size": (3, 5, 2),
                "stride": (2, 1, 1),
                "padding": (4, 2, 0),
            },
        )
        self.assertEqual(
            repr(conv3d_layer),
            repr(nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))),
        )

    def test_conv_transpose1d_creation(self):
        layer_factory = NNLayerFactory()
        conv_transpose1d_layer = layer_factory.get_layer(
            layer_type="Convolution",
            subtype="ConvTranspose1d",
            parameters={
                "in_channels": 3000,
                "out_channels": 3000,
                "kernel_size": (3, 3),
                "stride": 2,
                "padding": 5,
                "bias": False,
            },
        )
        self.assertEqual(
            repr(conv_transpose1d_layer),
            repr(
                nn.ConvTranspose1d(3000, 3000, (3, 3), stride=2, padding=5, bias=False)
            ),
        )

    def test_conv_transpose2d_creation(self):
        layer_factory = NNLayerFactory()
        conv_transpose2d_layer = layer_factory.get_layer(
            layer_type="Convolution",
            subtype="ConvTranspose2d",
            parameters={
                "in_channels": 3000,
                "out_channels": 3000,
                "kernel_size": 3,
                "stride": 2,
                "padding": 5,
                "bias": False,
            },
        )
        self.assertEqual(
            repr(conv_transpose2d_layer),
            repr(nn.ConvTranspose2d(3000, 3000, 3, stride=2, padding=5, bias=False)),
        )

    def test_conv_transpose3d_creation(self):
        layer_factory = NNLayerFactory()
        conv_transpose3d_layer = layer_factory.get_layer(
            layer_type="Convolution",
            subtype="ConvTranspose3d",
            parameters={
                "in_channels": 3000,
                "out_channels": 3000,
                "kernel_size": 3,
                "stride": 2,
                "padding": 5,
                "bias": False,
            },
        )
        self.assertEqual(
            repr(conv_transpose3d_layer),
            repr(nn.ConvTranspose3d(3000, 3000, 3, stride=2, padding=5, bias=False)),
        )

    def test_unfold_creation(self):
        layer_factory = NNLayerFactory()
        conv_unfold_layer = layer_factory.get_layer(
            layer_type="Convolution",
            subtype="Unfold",
            parameters={"kernel_size": (3, 5), "stride": 3, "dilation": 1},
        )
        self.assertEqual(
            repr(conv_unfold_layer),
            repr(nn.Unfold(kernel_size=(3, 5), stride=3, dilation=1)),
        )

    def test_fold_creation(self):
        layer_factory = NNLayerFactory()
        conv_fold_layer = layer_factory.get_layer(
            layer_type="Convolution",
            subtype="Fold",
            parameters={
                "output_size": (4, 5),
                "kernel_size": (2, 2),
                "stride": 3,
                "dilation": 1,
            },
        )
        self.assertEqual(
            repr(conv_fold_layer),
            repr(nn.Fold(output_size=(4, 5), kernel_size=(2, 2), stride=3, dilation=1)),
        )

    def test_fold_creation_with_extra_params(self):
        layer_factory = NNLayerFactory()
        conv_fold_layer = layer_factory.get_layer(
            layer_type="Convolution",
            subtype="Fold",
            parameters={
                "output_size": (4, 5),
                "kernel_size": (2, 2),
                "stride": 3,
                "dilation": 1,
                "invalid_param": (3, 4),
                "other_invalid_param": 34,
            },
        )
        self.assertEqual(
            repr(conv_fold_layer),
            repr(nn.Fold(output_size=(4, 5), kernel_size=(2, 2), stride=3, dilation=1)),
        )

    # Test with incorrect layer_type
    def test_conv1d_creation_invalid_layer_type(self):
        layer_factory = NNLayerFactory()
        with self.assertRaises(Exception, msg="Layer type not in options"):
            layer_factory.get_layer(
                layer_type="Convlution",
                subtype="Conv1d",
                parameters={
                    "in_channels": 16,
                    "out_channels": 33,
                    "kernel_size": 3,
                    "stride": 2,
                },
            )

    # Test with incorrect subtype for convolution
    def test_conv1d_creation_invalid_subtype_conv(self):
        layer_factory = NNLayerFactory()
        with self.assertRaises(
            Exception, msg="Convolutional layer type not in options"
        ):
            layer_factory.get_layer(
                layer_type="Convolution",
                subtype="Convo1d",
                parameters={
                    "in_channels": 16,
                    "out_channels": 33,
                    "kernel_size": 3,
                    "stride": 2,
                },
            )


if __name__ == "__main__":
    unittest.main()
