#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import numpy as np

from romiseg.train.datasets import DatasetImLabel, DatasetImLabel3d


class TestDatasetImLabelReadLabel(unittest.TestCase):

    def setUp(self):
        self.dataset = DatasetImLabel([], [], (1, 1), '')

    def test_adds_background_channel(self):
        """read_label prepends a background channel marking empty pixels."""
        labels = np.zeros((2, 4, 4), dtype=np.uint8)
        labels[0] = 1  # a foreground channel
        result = self.dataset.read_label(labels.copy())
        self.assertEqual(result.shape, (3, 4, 4))
        # Background is 255 exactly where every channel sums to zero.
        expected_bg = (labels.sum(axis=0) == 0).astype(np.uint8) * 255
        np.testing.assert_array_equal(result[0], expected_bg)
        # Original channels are preserved.
        np.testing.assert_array_equal(result[1], labels[0])
        np.testing.assert_array_equal(result[2], labels[1])

    def test_background_follows_labels(self):
        """Pixels covered by a foreground label are not part of the background."""
        labels = np.zeros((1, 3, 3), dtype=np.uint8)
        labels[0, 1, 1] = 255
        result = self.dataset.read_label(labels.copy())
        self.assertEqual(result[0, 1, 1], 0)  # covered pixel not background
        self.assertEqual(result[0, 0, 0], 255)  # empty pixel is background


class TestDatasetImLabel3dReadLabel(unittest.TestCase):

    def setUp(self):
        self.dataset = DatasetImLabel3d([], [], [], None)

    def test_adds_background_channel(self):
        """read_label prepends a 255 background layer over empty pixels."""
        labels = np.zeros((2, 3, 3), dtype=np.uint8)
        labels[0] = 5
        result = self.dataset.read_label(labels.copy())
        self.assertEqual(result.shape, (3, 3, 3))
        expected_bg = (labels.sum(axis=0) == 0).astype(np.uint8) * 255
        np.testing.assert_array_equal(result[0], expected_bg)
        np.testing.assert_array_equal(result[1], labels[0])


if __name__ == '__main__':
    unittest.main()
