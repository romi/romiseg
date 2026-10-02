#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import torch

from romiseg.train.metrics import my_metric, print_metrics


class TestMyMetric(unittest.TestCase):

    def test_perfect_match(self):
        """Accuracy is 1.0 when all non-zero labels match the outputs."""
        outputs = torch.tensor([[0., 1., 0.], [0., 1., 2.]])
        labels = torch.tensor([[0., 1., 0.], [0., 1., 2.]])
        self.assertAlmostEqual(my_metric(outputs, labels).item(), 1.0)

    def test_partial_match(self):
        """Accuracy reflects the fraction of matching non-zero labels."""
        outputs = torch.tensor([[1., 1., 0.]])
        labels = torch.tensor([[0., 1., 2.]])
        # Non-zero labels: 1 and 2. Only the first (1 == 1) matches.
        self.assertAlmostEqual(my_metric(outputs, labels).item(), 0.5)

    def test_zero_labels_excluded(self):
        """Labels equal to zero are excluded from the accuracy computation."""
        outputs = torch.tensor([[1., 1.]])
        labels = torch.tensor([[0., 1.]])
        # Only the last column is considered and it matches.
        self.assertAlmostEqual(my_metric(outputs, labels).item(), 1.0)

    def test_no_valid_labels_returns_nan(self):
        """When every label is zero the mean over an empty set is NaN."""
        outputs = torch.tensor([[1., 1.]])
        labels = torch.tensor([[0., 0.]])
        self.assertTrue(torch.isnan(my_metric(outputs, labels)))

    def test_print_metrics(self):
        """print_metrics computes the per-sample average of each metric."""
        import io as _io
        from contextlib import redirect_stdout
        metrics = {'acc': 20.0, 'loss': 5.0}
        buf = _io.StringIO()
        with redirect_stdout(buf):
            print_metrics(metrics, epoch_samples=2, phase='val')
        out = buf.getvalue()
        self.assertIn('val:', out)
        self.assertIn('acc: 10.000000', out)
        self.assertIn('loss: 2.500000', out)


if __name__ == '__main__':
    unittest.main()
