#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

import numpy as np

from romiseg.utils.ply import read_ply, write_ply


class TestWriteReadPly(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _path(self, name):
        return str(Path(self.tmp) / name)

    def test_roundtrip_points(self):
        """write_ply followed by read_ply preserves the point coordinates."""
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        path = self._path('points.ply')
        self.assertTrue(write_ply(path, [points], ['x', 'y', 'z']))
        data = read_ply(path)
        np.testing.assert_allclose(data['x'], points[:, 0])
        np.testing.assert_allclose(data['y'], points[:, 1])
        np.testing.assert_allclose(data['z'], points[:, 2])

    def test_roundtrip_multiple_fields(self):
        """Multiple fields and a 1D field are preserved through the round trip."""
        points = np.random.rand(5, 3)
        values = np.random.randint(0, 2, size=5).astype(np.float32)
        path = self._path('multi.ply')
        self.assertTrue(write_ply(path, [points, values], ['x', 'y', 'z', 'v']))
        data = read_ply(path)
        np.testing.assert_allclose(np.vstack((data['x'], data['y'], data['z'])).T, points)
        np.testing.assert_allclose(data['v'], values)

    def test_extension_added(self):
        """The '.ply' extension is appended when missing from the filename."""
        path = self._path('noext')
        self.assertTrue(write_ply(path, [np.random.rand(2, 3)], ['x', 'y', 'z']))
        self.assertTrue(Path(path + '.ply').exists())

    def test_mismatched_field_names_returns_false(self):
        """write_ply returns False when the number of field names is wrong."""
        path = self._path('bad.ply')
        self.assertFalse(write_ply(path, [np.random.rand(2, 3)], ['x']))

    def test_rejects_non_ply_file(self):
        """read_ply raises ValueError for a file that does not start with 'ply'."""
        path = self._path('bad.ply')
        with open(path, 'wb') as f:
            f.write(b'not a valid header\n')
        with self.assertRaises(ValueError):
            read_ply(path)


if __name__ == '__main__':
    unittest.main()
