#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import torch
import torch.nn as nn

from romiseg.predict import evaluation
from romiseg.predict.evaluation import evaluate

eval_test = evaluation.test


def _tiny_model():
    """Build a tiny segmentation model to avoid loading a full U-Net."""
    return nn.Sequential(nn.Conv2d(3, 2, 1))


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        self.model = _tiny_model()
        self.inputs = torch.rand(2, 3, 8, 8)

    def test_output_shape(self):
        """evaluate returns a tensor matching the batch and class dimensions."""
        pred = evaluate(self.inputs, self.model)
        self.assertEqual(pred.shape, (2, 2, 8, 8))

    def test_output_uses_sigmoid(self):
        """The raw logits are passed through a sigmoid, bounding them to [0, 1]."""
        pred = evaluate(self.inputs, self.model)
        self.assertGreaterEqual(pred.min().item(), 0.0)
        self.assertLessEqual(pred.max().item(), 1.0)

    def test_no_grad(self):
        """evaluate runs under torch.no_grad so no gradients are tracked."""
        pred = evaluate(self.inputs, self.model)
        self.assertFalse(pred.requires_grad)


class TestTest(unittest.TestCase):

    def setUp(self):
        self.model = _tiny_model()
        self.inputs = torch.rand(2, 3, 8, 8)
        self.labels = (torch.rand(2, 2, 8, 8) > 0.5).float()

    def test_returns_predictions_and_metrics(self):
        """test returns the sigmoid predictions and a metrics dictionary."""
        pred, metrics = eval_test(self.inputs, self.labels, self.model)
        self.assertEqual(pred.shape, (2, 2, 8, 8))
        self.assertIn('bce', metrics)
        self.assertIn('dice', metrics)
        self.assertIn('loss', metrics)

    def test_metrics_are_positive(self):
        """The accumulated losses are non-negative floats."""
        _, metrics = eval_test(self.inputs, self.labels, self.model)
        for name in ('bce', 'dice', 'loss'):
            self.assertGreaterEqual(float(metrics[name]), 0.0)


if __name__ == '__main__':
    unittest.main()
