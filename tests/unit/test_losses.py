#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import torch

from romiseg.train.losses import calc_loss, dice_loss


class TestDiceLoss(unittest.TestCase):

    def test_perfect_overlap_zero_loss(self):
        """Dice loss is zero when prediction and target fully overlap."""
        pred = torch.ones(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        self.assertAlmostEqual(dice_loss(pred, target).item(), 0.0, places=4)

    def test_no_overlap_high_loss(self):
        """Dice loss is near one when prediction and target do not overlap."""
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        loss = dice_loss(pred, target).item()
        self.assertGreater(loss, 0.9)
        self.assertLess(loss, 1.0)

    def test_averaged_over_batch(self):
        """The returned loss is the mean over the batch dimension."""
        pred = torch.ones(2, 1, 4, 4)
        target = torch.ones(2, 1, 4, 4)
        self.assertAlmostEqual(dice_loss(pred, target).item(), 0.0, places=4)

    def test_partial_overlap_between_zero_and_one(self):
        """Partial overlap yields a Dice loss strictly between 0 and 1."""
        pred = torch.ones(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        target[..., 0] = 0  # one column of the target is empty
        loss = dice_loss(pred, target).item()
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 1.0)


class TestCalcLoss(unittest.TestCase):

    def test_updates_metrics_in_place(self):
        """calc_loss accumulates the BCE, Dice and combined losses into metrics."""
        torch.manual_seed(0)
        pred = torch.randn(2, 1, 8, 8)
        target = (torch.rand(2, 1, 8, 8) > 0.5).float()
        metrics = {'bce': 0.0, 'dice': 0.0, 'loss': 0.0}
        loss = calc_loss(pred, target, metrics)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        # Each metric is multiplied by the batch size (2).
        self.assertAlmostEqual(metrics['loss'] / 2, loss.item(), places=5)
        self.assertGreater(metrics['bce'], 0.0)
        self.assertGreater(metrics['dice'], 0.0)

    def test_bce_weight(self):
        """The returned loss is the weighted combination of BCE and Dice."""
        pred = torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        metrics = {'bce': 0.0, 'dice': 0.0, 'loss': 0.0}
        loss = calc_loss(pred, target, metrics, bce_weight=1.0)
        # With bce_weight=1.0 the loss is exactly the BCE loss.
        expected = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)


if __name__ == '__main__':
    unittest.main()
