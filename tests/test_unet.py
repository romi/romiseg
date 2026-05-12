#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import torch

from romiseg.models.unet import ResNetUNet


class TestResNetUNet(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create dummy input tensor with shape (batch_size, channels, height, width)
        self.dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
        # Initialize model and move to appropriate device
        self.n_classes = 2
        self.model = ResNetUNet(self.n_classes).to(self.device)

    def test_model_initialization(self):
        """Test that the model initializes with required attributes."""
        expected_attributes = [
            'conv_up0', 'base_model', 'conv_up1', 'conv_up2',
            'layer0', 'conv_up3', 'conv_original_size2',
            'conv_original_size1', 'conv_original_size0',
            'layer0_1x1', 'layer2_1x1', 'conv_last',
            'base_layers', 'upsample', 'layer4_1x1',
            'layer3', 'layer4', 'layer1', 'layer2',
            'layer3_1x1', 'layer1_1x1'
        ]
        for attr in expected_attributes:
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(self.model, attr),
                    f"Model is missing required attribute: {attr}"
                )

    def test_forward_pass_shape(self):
        """Test that the forward pass produces output of correct shape."""
        with torch.no_grad():
            output = self.model(self.dummy_input)
            # Adjust expected_shape based on your model's actual output
            expected_shape = (1, self.n_classes, 64, 64)  # Example for binary segmentation mask
            self.assertEqual(
                output.shape,
                expected_shape,
                f"Expected output shape {expected_shape}, got {output.shape}"
            )

    def test_forward_pass_output_finite(self):
        """Test that the forward pass produces finite outputs."""
        with torch.no_grad():
            output = self.model(self.dummy_input)
            self.assertFalse(torch.isnan(output).any(),
                             "Output contains NaN values")
            self.assertFalse(torch.isinf(output).any(),
                             "Output contains infinite values")


if __name__ == '__main__':
    unittest.main()
