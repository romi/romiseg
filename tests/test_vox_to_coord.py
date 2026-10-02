#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import torch

from romiseg.utils.vox_to_coord import avoid_eps, basis_vox, get_extr, get_int


class TestAvoidEps(unittest.TestCase):

    def test_small_values_zeroed(self):
        """Elements with absolute value below eps are set to zero."""
        a = torch.tensor([0.0, 0.5, -0.2, 1.0])
        result = avoid_eps(a.clone(), eps=0.3)
        np_result = result.numpy()
        self.assertEqual(np_result[0], 0.0)
        self.assertAlmostEqual(np_result[1], 0.5)
        self.assertEqual(np_result[2], 0.0)
        self.assertAlmostEqual(np_result[3], 1.0)


class TestBasisVox(unittest.TestCase):

    def test_shape_and_count(self):
        """basis_vox returns a (w*h*l, 4) array."""
        voxels = basis_vox([0, 0, 0], w=2, h=3, l=4)
        self.assertEqual(voxels.shape, (24, 4))
        self.assertTrue((voxels[:, 3] == 0).all())

    def test_coordinate_ranges(self):
        """X, Y and Z take the expected distinct values."""
        voxels = basis_vox([1, 2, 3], w=2, h=3, l=4)
        self.assertEqual(sorted(set(voxels[:, 0].tolist())), [1, 2])
        self.assertEqual(sorted(set(voxels[:, 1].tolist())), [2, 3, 4])
        self.assertEqual(sorted(set(voxels[:, 2].tolist())), [3, 4, 5, 6])


class TestGetInt(unittest.TestCase):

    def test_intrinsic_matrix(self):
        """get_int returns the standard 3x3 intrinsic matrix."""
        m = get_int(100.0, 200.0, 320.0, 240.0)
        self.assertEqual(m.shape, (3, 3))
        self.assertEqual(m[0, 0].item(), 100.0)
        self.assertEqual(m[1, 1].item(), 200.0)
        self.assertEqual(m[0, 2].item(), 320.0)
        self.assertEqual(m[1, 2].item(), 240.0)
        self.assertEqual(m[2, 2].item(), 1.0)


class TestGetExtr(unittest.TestCase):

    def test_shape(self):
        """get_extr returns a 3x4 extrinsic matrix."""
        m = get_extr(0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
        self.assertEqual(m.shape, (3, 4))

    def test_zero_rotation_translation(self):
        """With zero rotation the translation column reflects the input offset."""
        m = get_extr(0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
        self.assertAlmostEqual(m[0, 3].item(), -1.0)
        self.assertAlmostEqual(m[1, 3].item(), -2.0)
        self.assertAlmostEqual(m[2, 3].item(), -3.0)


if __name__ == '__main__':
    unittest.main()
