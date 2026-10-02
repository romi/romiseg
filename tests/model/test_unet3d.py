#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from romiseg.models.unet3d import ResNetUNet_3D, Classification

class TestResNetUNet_3D(unittest.TestCase):
    def setUp(self):
        # Initialize a basic instance of ResNetUNet_3D with minimal parameters for testing
        self.model = ResNetUNet_3D(n_class=2)

    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, ResNetUNet_3D)  # Check if instance is created

    def test_forward_pass(self):
        """Test the forward pass of the network with random input."""
        import torch
        x = torch.randn(1, 1, 64, 64, 64)  # Random input tensor
        output = self.model.forward(x)
        self.assertIsNotNone(output)  # Check if output is generated

    def test_output_shape(self):
        """Test the shape of the output after forward pass."""
        import torch
        x = torch.randn(1, 1, 64, 64, 64)  # Random input tensor
        output = self.model.forward(x)
        expected_shape = (1, 2, 64, 64, 64)  # Assuming same spatial dimensions and n_class=2
        self.assertEqual(output.shape, expected_shape)

class TestClassification(unittest.TestCase):
    def setUp(self):
        import torch.nn as nn
        self.layer = nn.Conv3d(1, 2, kernel_size=3, padding=1)  # Example convolution layer
        D_out = 4  # Define D_out for the Classification class
        self.classification = Classification(self.layer, D_out)

    def test_initialization(self):
        """Test that the classification module initializes correctly."""
        self.assertIsInstance(self.classification, Classification)

    def test_forward_pass(self):
        """Test the forward pass of the classification module with random input."""
        import torch
        x = torch.randn(1, 1, 64, 64, 64)  # Random input tensor
        output = self.classification.forward(x)
        self.assertIsNotNone(output)

    def test_output_shape(self):
        """Test the shape of the output after forward pass."""
        import torch
        x = torch.randn(1, 1, 64, 64, 64)  # Random input tensor
        output = self.classification.forward(x)
        expected_shape = (1, 2, 64, 64, 64)  # Assuming same spatial dimensions and n_class=2
        self.assertEqual(output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()