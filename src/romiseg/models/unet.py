#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D UNet model definitions for the romiseg package.

This module contains the 2D UNet model definitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def convrelu(in_channels, out_channels, kernel, padding):
    """Creates a sequential container with a convolutional layer followed by a ReLU activation function.

    The function constructs and returns a `torch.nn.Sequential` object, which
    includes a 2D convolutional layer and a ReLU activation. It allows defining
    a quick combination of convolution and non-linearity without having to
    manually instantiate and arrange layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the convolutional layer.
    out_channels : int
        Number of output channels produced by the convolutional layer.
    kernel : int or Tuple[int, int]
        Size of the convolving kernel. Can be specified as a single integer
        (for square kernels) or a tuple (for rectangular kernels).
    padding : int
        Amount of zero-padding added to all sides of the input.

    Returns
    -------
    torch.nn.Sequential
        A sequential container consisting of a convolutional layer followed by
        a ReLU activation function in order.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    """ResNetUNet is a U-Net-based deep convolutional neural network designed for semantic segmentation tasks.

    It combines the ResNet101 architecture as an encoder with a decoder to perform pixel-wise predictions.
    This network features skip connections between the encoder and the decoder to preserve spatial information.
    It employs a series of convolutional and upsampling layers to restore the input resolution for segmentation output.

    Attributes
    ----------
    base_model : torch.nn.Module
        Pretrained ResNet101 model used as the encoder backbone.
    base_layers : list
        List of individual layers extracted from the pretrained ResNet model.
    layer0 : torch.nn.Sequential
        First three convolutional layers of the encoder, downsampled by a factor of 2.
    layer0_1x1 : torch.nn.Module
        1x1 convolutional layer applied after `layer0` to adjust the number of channels.
    layer1 : torch.nn.Sequential
        Encoder layers encompassing ResNet's next block, reducing the resolution by a factor of 4.
    layer1_1x1 : torch.nn.Module
        1x1 convolutional layer applied after `layer1` to adjust the number of channels.
    layer2 : torch.nn.Module
        Encoder layers reducing the resolution to a factor of 8.
    layer2_1x1 : torch.nn.Module
        1x1 convolutional layer applied after `layer2` to adjust the number of channels.
    layer3 : torch.nn.Module
        Encoder layers reducing the resolution to a factor of 16.
    layer3_1x1 : torch.nn.Module
        1x1 convolutional layer applied after `layer3` to adjust the number of channels.
    layer4 : torch.nn.Module
        Final encoder block reducing the resolution to a factor of 32.
    layer4_1x1 : torch.nn.Module
        1x1 convolutional layer applied after `layer4` to adjust the number of channels.
    upsample : torch.nn.Upsample
        Bilinear upsampling layer used to increase spatial resolution by a factor of 2.
    conv_up3 : torch.nn.Module
        3x3 convolutional layer used for processing concatenated features from layer3 and decoder path.
    conv_up2 : torch.nn.Module
        3x3 convolutional layer used for processing concatenated features from layer2 and decoder path.
    conv_up1 : torch.nn.Module
        3x3 convolutional layer used for processing concatenated features from layer1 and decoder path.
    conv_up0 : torch.nn.Module
        3x3 convolutional layer used for processing concatenated features from layer0 and decoder path.
    conv_original_size0 : torch.nn.Module
        Initial 3x3 convolutional layer to process the original input size.
    conv_original_size1 : torch.nn.Module
        Intermediate 3x3 convolutional layer to further process the original input.
    conv_original_size2 : torch.nn.Module
        Final 3x3 convolutional layer to fuse the output with the processed original input.
    conv_last : torch.nn.Conv2d
        Final 1x1 convolutional layer to produce the segmentation map with `n_class` output channels.
    """

    def __init__(self, n_class):
        super().__init__()

        # Use ResNet101 as the encoder with the pretrained weights
        self.base_model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 256, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(256, 256, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(512, 512, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(1024, 512, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(2048, 1024, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up2 = convrelu(512 + 512, 512, 3, 1)
        self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        """Perform the forward pass through the model for generating predictions based on input.

        Processes an input tensor through the network layers, performing a series of convolution,
        upsampling, and concatenation operations to extract and merge multi-level features.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor to the forward pass, typically of dimensions
            (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            The output tensor after processing through the network. This tensor typically
            represents the final feature map or the reconstructed output.
        """
        x_original = self.conv_original_size0(input)
        # print(x_original.shape)
        x_original = self.conv_original_size1(x_original)
        # print(x_original.shape)
        layer0 = self.layer0(input)
        # print(layer0.shape)
        layer1 = self.layer1(layer0)
        # print(layer1.shape)
        layer2 = self.layer2(layer1)
        # print(layer2.shape)
        layer3 = self.layer3(layer2)
        # print(layer3.shape)
        layer4 = self.layer4(layer3)
        # print(layer4.shape)
        # Upsample the last/bottom layer
        layer4 = self.layer4_1x1(layer4)
        # print(layer4.shape)

        # x = self.upsample(layer4)  # old API
        x = F.interpolate(layer4, scale_factor=2, mode='bilinear', align_corners=False)  # new API

        # print(x.shape)
        # Create the shortcut from the encoder
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        # print(x.shape)

        x = self.conv_up3(x)
        # print(x.shape)

        # x = self.upsample(x)  # old API
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # new API
        # print(x.shape)

        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        # print(x.shape)

        x = self.conv_up2(x)
        # print(x.shape)

        # x = self.upsample(x)  # old API
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # new API
        # print(x.shape)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        # print(x.shape)
        x = self.conv_up1(x)
        # print(x.shape)

        # x = self.upsample(x)  # old API
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # new API
        # print(x.shape)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        # print(x.shape)
        x = self.conv_up0(x)
        # print(x.shape)

        # x = self.upsample(x)  # old API
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # new API
        # print(x.shape)
        x = torch.cat([x, x_original], dim=1)
        # print(x.shape)
        x = self.conv_original_size2(x)
        # print(x.shape)

        out = self.conv_last(x)

        return out