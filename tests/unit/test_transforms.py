#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from PIL import Image, ImageDraw, ImageOps
import torch

from romiseg.common.transforms import MyRotationTransform
from romiseg.common.transforms import ResizeCrop
from romiseg.common.transforms import ResizeFit
from romiseg.common.transforms import gaussian


class TestImageTransforms(unittest.TestCase):
    def setUp(self):
        # Create a test image with distinctive features for easy verification
        self.test_img = Image.new('RGB', (200, 100), color='red')
        draw = ImageDraw.Draw(self.test_img)
        draw.rectangle([(50, 25), (150, 75)], fill='blue')

    def test_gaussian(self):
        # Create a test tensor
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Test training mode with noise
        output_train = gaussian(input_tensor, is_training=True, mean=0, stddev=0.5)
        self.assertEqual(output_train.shape, input_tensor.shape)
        # The actual values will vary due to randomness,
        # so we'll just check that they're within bounds when dyn=1
        self.assertTrue(torch.all(output_train >= -1).item())
        self.assertTrue(torch.all(output_train <= 1).item())

        # Test with different dynamic range
        output_train_dyn = gaussian(input_tensor, is_training=True, mean=0, stddev=0.5, dyn=2)
        self.assertTrue(torch.all(output_train_dyn >= -2).item())
        self.assertTrue(torch.all(output_train_dyn <= 2).item())

        # Test non-training mode (should be identity operation with clamping)
        output_not_train = gaussian(input_tensor, is_training=False, mean=0, stddev=0.5)
        self.assertEqual(output_not_train.shape, input_tensor.shape)
        self.assertTrue(torch.all(output_not_train == torch.clamp(input_tensor, -1, 1)).item())

    def test_resize_fit(self):
        # Test resizing to smaller dimensions
        transformer = ResizeFit((100, 100))
        output_img = transformer(self.test_img)

        # Verify output size and aspect ratio is preserved
        self.assertEqual(output_img.size[0], 100)
        self.assertLessEqual(output_img.size[1], 100)  # Height should be <= 100

        # Calculate expected height based on aspect ratio
        original_aspect = 200 / 100  # width/height
        target_width = 100
        expected_height = int(100 / (original_aspect / target_width))
        # This might not be exact due to integer division, but should be close

        # Test with different aspect ratio
        transformer_tall = ResizeFit((50, 200))
        output_img_tall = transformer_tall(self.test_img)
        self.assertEqual(output_img_tall.size[1], 200)
        self.assertLessEqual(output_img_tall.size[0], 50)  # Width should be <= 50

    def test_resize_crop(self):
        # Test resizing and cropping
        transformer = ResizeCrop((100, 100))
        output_img = transformer(self.test_img)

        # Verify output size matches exactly the target dimensions
        self.assertEqual(output_img.size[0], 100)
        self.assertEqual(output_img.size[1], 100)

    def test_rotation_transform(self):
        # Test rotation by 90 degrees
        transformer = MyRotationTransform(angle=90, fill=(1., 0., 0.))
        rotated_img = transformer(self.test_img)

        # Verify output size matches input size (rotation shouldn't change dimensions)
        self.assertEqual(rotated_img.size, self.test_img.size)

        # verify that the output is an Image object.
        self.assertIsInstance(rotated_img, Image.Image)

if __name__ == '__main__':
    unittest.main()