import unittest
import torch
import torch.nn as nn

import sys


sys.path.insert(0, "../../")
sys.path.insert(1, "../../nn_model")
sys.path.insert(2, "../../nn_loss")

from nn_loss.nn_loss_factory import NNLossFactory


class TestLossFactory(unittest.TestCase):
    def test_l1loss_creation(self):
        loss_factory = NNLossFactory()
        l1loss = loss_factory.get_loss(
            "L1Loss", {"size_average": True, "reduce": False, "reduction": "sum"}
        )

        self.assertEqual(
            repr(l1loss),
            repr(torch.nn.L1Loss(size_average=True, reduce=False, reduction="sum")),
        )

    def test_l1loss_with_none_reduction_creation(self):
        loss_factory = NNLossFactory()
        l1loss = loss_factory.get_loss("L1Loss", {"reduction": "none"})

        self.assertEqual(
            repr(l1loss),
            repr(torch.nn.L1Loss(reduction="none")),
        )

    def test_mseloss_creation(self):
        loss_factory = NNLossFactory()
        mseloss = loss_factory.get_loss(
            "MSELoss", {"size_average": True, "reduce": False, "reduction": "sum"}
        )

        self.assertEqual(
            repr(mseloss),
            repr(torch.nn.MSELoss(size_average=True, reduce=False, reduction="sum")),
        )

    def test_crossentropyloss_creation(self):
        loss_factory = NNLossFactory()
        weight = torch.tensor([[1]])
        crossentropyloss = loss_factory.get_loss(
            "CrossEntropyLoss", {"weight": weight, "ignore_index": 0}
        )

        self.assertEqual(
            repr(crossentropyloss),
            repr(torch.nn.CrossEntropyLoss(weight=weight, ignore_index=0)),
        )

    def test_nllloss_creation(self):
        loss_factory = NNLossFactory()
        weight = torch.tensor([[1]])
        nllloss = loss_factory.get_loss(
            "NLLLoss", {"weight": weight, "ignore_index": 0}
        )

        self.assertEqual(
            repr(nllloss),
            repr(torch.nn.NLLLoss(weight=weight, ignore_index=0)),
        )

    def test_poissonnllloss_creation(self):
        loss_factory = NNLossFactory()
        poissonnllloss = loss_factory.get_loss(
            "PoissonNLLLoss", {"log_input": True, "full": False, "eps": 1e-7}
        )

        self.assertEqual(
            repr(poissonnllloss),
            repr(torch.nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-7)),
        )

    def test_gaussiannllloss_creation(self):
        loss_factory = NNLossFactory()
        gaussiannllloss = loss_factory.get_loss(
            "GaussianNLLLoss", {"full": False, "eps": 1e-7}
        )

        self.assertEqual(
            repr(gaussiannllloss),
            repr(torch.nn.GaussianNLLLoss(full=True, eps=1e-7)),
        )

    def test_bceloss_creation(self):
        loss_factory = NNLossFactory()
        bceloss = loss_factory.get_loss("BCELoss", {"reduction": "sum"})

        self.assertEqual(
            repr(bceloss),
            repr(torch.nn.BCELoss(reduction="sum")),
        )

    def test_bcewithlogitsloss_creation(self):
        loss_factory = NNLossFactory()
        pos_weight = torch.tensor([[1]])
        bcewithlogitsloss = loss_factory.get_loss(
            "BCEWithLogitsLoss", {"reduction": "sum", "pos_weight": pos_weight}
        )

        self.assertEqual(
            repr(bcewithlogitsloss),
            repr(torch.nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)),
        )

    def test_softmarginloss_creation(self):
        loss_factory = NNLossFactory()
        softmarginloss = loss_factory.get_loss("SoftMarginLoss", {"reduction": "mean"})

        self.assertEqual(
            repr(softmarginloss), repr(torch.nn.SoftMarginLoss(reduction="mean"))
        )

    def test_multilabelsoftmarginloss_creation(self):
        loss_factory = NNLossFactory()
        multilabelsoftmarginloss = loss_factory.get_loss(
            "MultiLabelSoftMarginLoss", {"reduction": "mean"}
        )

        self.assertEqual(
            repr(multilabelsoftmarginloss),
            repr(torch.nn.MultiLabelSoftMarginLoss(reduction="mean")),
        )
