import unittest
import torch
import torch.nn as nn

import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model")
from nn_model.nn_layer_factory import NNLayerFactory


class TestPoolingLayerFactory(unittest.TestCase):
    def test_maxpool1d_creation(self):
        layer_factory = NNLayerFactory()
        maxpool1d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="MaxPool1d",
            parameters={
                "kernel_size": 3,
                "stride": 2,
            },
        )
        self.assertEqual(repr(maxpool1d_layer), repr(nn.MaxPool1d(3, stride=2)))

    def test_maxpool1d_creation_with_extra_params(self):
        layer_factory = NNLayerFactory()
        maxpool1d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="MaxPool1d",
            parameters={
                "input_size": (3, 4),
                "kernel_size": 3,
                "stride": 2,
            },
        )
        self.assertEqual(repr(maxpool1d_layer), repr(nn.MaxPool1d(3, stride=2)))

    def test_maxpool2d_creation(self):
        layer_factory = NNLayerFactory()
        maxpool2d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="MaxPool2d",
            parameters={
                "kernel_size": (3, 2),
                "stride": (2, 1),
            },
        )
        self.assertEqual(
            repr(maxpool2d_layer), repr(nn.MaxPool2d((3, 2), stride=(2, 1)))
        )

    def test_maxpool3d_creation(self):
        layer_factory = NNLayerFactory()
        maxpool3d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="MaxPool3d",
            parameters={
                "kernel_size": (3, 2, 2),
                "stride": (2, 1, 2),
            },
        )
        self.assertEqual(
            repr(maxpool3d_layer), repr(nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2)))
        )

    def test_maxunpool1d_creation(self):
        layer_factory = NNLayerFactory()
        max_unpool1d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="MaxUnpool1d",
            parameters={
                "kernel_size": 2,
                "stride": 2,
            },
        )
        self.assertEqual(repr(max_unpool1d_layer), repr(nn.MaxUnpool1d(2, stride=2)))

    def test_maxunpool2d_creation(self):
        layer_factory = NNLayerFactory()
        max_unpool2d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="MaxUnpool2d",
            parameters={
                "kernel_size": 2,
                "stride": 2,
            },
        )
        self.assertEqual(repr(max_unpool2d_layer), repr(nn.MaxUnpool2d(2, stride=2)))

    def test_maxunpool3d_creation(self):
        layer_factory = NNLayerFactory()
        max_unpool3d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="MaxUnpool3d",
            parameters={"kernel_size": 2, "stride": 2},
        )
        self.assertEqual(
            repr(max_unpool3d_layer),
            repr(nn.MaxUnpool3d(2, stride=2)),
        )

    def test_avgpool1d_creation(self):
        layer_factory = NNLayerFactory()
        avgpool1d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="AvgPool1d",
            parameters={"kernel_size": 3, "stride": 2},
        )
        self.assertEqual(repr(avgpool1d_layer), repr(nn.AvgPool1d(3, stride=2)))

    def test_avgpool2d_creation(self):
        layer_factory = NNLayerFactory()
        avgpool2d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="AvgPool2d",
            parameters={"kernel_size": 3, "stride": 2},
        )
        self.assertEqual(repr(avgpool2d_layer), repr(nn.AvgPool2d(3, stride=2)))

    def test_avgpool3d_creation(self):
        layer_factory = NNLayerFactory()
        avgpool3d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="AvgPool3d",
            parameters={"kernel_size": 3, "stride": 2},
        )
        self.assertEqual(repr(avgpool3d_layer), repr(nn.AvgPool3d(3, stride=2)))

    def test_fractional_maxpool2d_creation(self):
        layer_factory = NNLayerFactory()
        fractional_maxpool2d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="FractionalMaxPool2d",
            parameters={"kernel_size": 3, "output_ratio": (0.5, 0.5)},
        )
        self.assertEqual(
            repr(fractional_maxpool2d_layer),
            repr(nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))),
        )

    def test_fractional_maxpool3d_creation(self):
        layer_factory = NNLayerFactory()
        fractional_maxpool3d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="FractionalMaxPool3d",
            parameters={"kernel_size": 3, "output_ratio": (0.5, 0.5, 0.5)},
        )
        self.assertEqual(
            repr(fractional_maxpool3d_layer),
            repr(nn.FractionalMaxPool3d(3, output_ratio=(0.5, 0.5, 0.5))),
        )

    def test_lppool1d_creation(self):
        layer_factory = NNLayerFactory()
        lppool1d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="LPPool1d",
            parameters={"kernel_size": 3, "stride": 2, "norm_type": 2},
        )
        self.assertEqual(
            repr(lppool1d_layer),
            repr(nn.LPPool1d(2, 3, stride=2)),
        )

    def test_lppool1d_creation_without_norm_type(self):
        layer_factory = NNLayerFactory()
        with self.assertRaises(Exception, msg="Norm type is required for LPPool1d"):
            layer_factory.get_layer(
                layer_type="Pooling",
                subtype="LPPool1d",
                parameters={"kernel_size": 3, "stride": 2},
            )

    def test_lppool2d_creation(self):
        layer_factory = NNLayerFactory()
        lppool2d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="LPPool2d",
            parameters={"kernel_size": 3, "stride": 2, "norm_type": 2},
        )
        self.assertEqual(
            repr(lppool2d_layer),
            repr(nn.LPPool2d(2, 3, stride=2)),
        )

    def test_lppool2d_creation_without_norm_type(self):
        layer_factory = NNLayerFactory()
        with self.assertRaises(Exception, msg="Norm type is required for LPPool2d"):
            layer_factory.get_layer(
                layer_type="Pooling",
                subtype="LPPool2d",
                parameters={"kernel_size": 3, "stride": 2},
            )

    def test_adaptive_maxpool1d_creation(self):
        layer_factory = NNLayerFactory()
        adaptive_maxpool1d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="AdaptiveMaxPool1d",
            parameters={"output_size": 5},
        )
        self.assertEqual(
            repr(adaptive_maxpool1d_layer),
            repr(nn.AdaptiveMaxPool1d(5)),
        )

    def test_adaptive_maxpool2d_creation(self):
        layer_factory = NNLayerFactory()
        adaptive_maxpool2d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="AdaptiveMaxPool2d",
            parameters={"output_size": 5},
        )
        self.assertEqual(
            repr(adaptive_maxpool2d_layer),
            repr(nn.AdaptiveMaxPool2d(5)),
        )

    def test_adaptive_maxpool3d_creation(self):
        layer_factory = NNLayerFactory()
        adaptive_maxpool3d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="AdaptiveMaxPool3d",
            parameters={"output_size": 5},
        )
        self.assertEqual(
            repr(adaptive_maxpool3d_layer),
            repr(nn.AdaptiveMaxPool3d(5)),
        )

    def test_adaptive_avgpool1d_creation(self):
        layer_factory = NNLayerFactory()
        adaptive_avgpool1d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="AdaptiveAvgPool1d",
            parameters={"output_size": 5},
        )
        self.assertEqual(
            repr(adaptive_avgpool1d_layer),
            repr(nn.AdaptiveAvgPool1d(5)),
        )

    def test_adaptive_avgpool2d_creation(self):
        layer_factory = NNLayerFactory()
        adaptive_avgpool2d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="AdaptiveAvgPool2d",
            parameters={"output_size": 5},
        )
        self.assertEqual(
            repr(adaptive_avgpool2d_layer),
            repr(nn.AdaptiveAvgPool2d(5)),
        )

    def test_adaptive_avgpool3d_creation(self):
        layer_factory = NNLayerFactory()
        adaptive_avgpool3d_layer = layer_factory.get_layer(
            layer_type="Pooling",
            subtype="AdaptiveAvgPool3d",
            parameters={"output_size": 5},
        )
        self.assertEqual(
            repr(adaptive_avgpool3d_layer),
            repr(nn.AdaptiveAvgPool3d(5)),
        )

    # Test with incorrect layer_type
    def test_maxpool1d_creation_invalid_layer_type(self):
        layer_factory = NNLayerFactory()
        with self.assertRaises(Exception, msg="Layer type not in options"):
            layer_factory.get_layer(
                layer_type="Poling",
                subtype="MaxPool1d",
                parameters={
                    "kernel_size": 3,
                    "stride": 2,
                },
            )

    # Test with incorrect subtype for pooling
    def test_maxpool1dcreation_invalid_subtype_conv(self):
        layer_factory = NNLayerFactory()
        with self.assertRaises(Exception, msg="Pooling layer type not in options"):
            layer_factory.get_layer(
                layer_type="Pooling",
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
