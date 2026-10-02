#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image
import torch

from romiseg.predict.datasets import DatasetImId


class TestDatasetImId(unittest.TestCase):

    def setUp(self):
        """Set up some test data for use in tests."""
        self.image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
        self.transforms = MagicMock(return_value=torch.rand(3, 64, 64))  # Mock transformation

    def test_initialization(self):
        """Test that the DatasetImId class initializes correctly."""
        dataset = DatasetImId(image_paths=self.image_paths, transform=self.transforms)
        self.assertEqual(dataset.image_paths, self.image_paths)
        self.assertEqual(dataset.transforms, self.transforms)

    @patch('PIL.Image.fromarray')
    def test_getitem(self, mock_fromarray):
        """Test the __getitem__ method of DatasetImId."""
        # Create a mock image
        mock_image = MagicMock()
        mock_fromarray.return_value = mock_image

        # Setup mock for reading images
        with patch('plantdb.commons.io.read_image', return_value=torch.rand(64, 64, 3)):
            dataset = DatasetImId(image_paths=self.image_paths, transform=self.transforms)

            t_image, image_id = dataset[0]

            # Check the transformation is applied correctly
            self.transforms.assert_called_once_with(mock_image)
            self.assertEqual(t_image.shape, (3, 64, 64))  # Transform should return a tensor of shape (C, H, W)
            self.assertEqual(image_id, Path(self.image_paths[0]).name)

    def test_getitem_empty_dataset(self):
        """Test that accessing an empty dataset raises an IndexError."""
        dataset = DatasetImId(image_paths=[], transform=self.transforms)
        with self.assertRaises(IndexError):
            dataset[0]

    @patch('PIL.Image.fromarray')
    def test_len(self, mock_fromarray):
        """Test the __len__ method of DatasetImId."""
        # Create a mock image
        mock_image = MagicMock()
        mock_fromarray.return_value = mock_image

        with patch('plantdb.commons.io.read_image', return_value=torch.rand(64, 64, 3)):
            dataset = DatasetImId(image_paths=self.image_paths, transform=self.transforms)
            self.assertEqual(len(dataset), len(self.image_paths))

if __name__ == '__main__':
    unittest.main()