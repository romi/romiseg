#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D UNet model definitions for the romiseg package.

This module contains the 3D UNet model definitions and related utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import romiseg.utils.vox_to_coord as vtc
from romiseg.models.unet import convrelu

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def voxel_to_pred_by_project(the_shape, torch_voxels, intrinsics, extrinsics, preds_flat, pred_pad, Sx, Sy, xinit,
                             yinit):
    """
    Projects voxel coordinates onto a 2D grid, assigns predictions to the voxels, and updates voxel data
    based on aggregated predictions.

    The method takes voxel coordinates and projects them onto 2D image space using given camera intrinsics
    and extrinsics. It corrects coordinate values that project out of the field of view. Predicted values
    are then assigned to voxels using the flattened 2D coordinates, aggregated, and used to update the voxel
    properties.

    Parameters
    ----------
    the_shape : tuple
        Shape of the 2D grid (height, width) used for flattening and assigning predictions.
    torch_voxels : Tensor
        Tensor of voxel data containing spatial coordinates and properties for each voxel.
    intrinsics : Tensor
        Camera intrinsic parameters used for projecting the voxel coordinates onto image space.
    extrinsics : Tensor
        Camera extrinsic parameters used for transforming the voxel coordinates into camera space.
    preds_flat : Tensor
        Flattened 2D tensor of predictions, where each pixel corresponds to prediction indices.
    pred_pad : Tensor
        Padding-related tensor used to reshape prediction assignments.
    Sx : int
        Scaling factor for the x-axis to account for grid transformations or corrections.
    Sy : int
        Scaling factor for the y-axis to account for grid transformations or corrections.
    xinit : int
        Offset value along the x-axis to adjust the projected coordinates.
    yinit : int
        Offset value along the y-axis to adjust the projected coordinates.

    Returns
    -------
    Tensor
        Updated tensor of voxel data where the fourth property of each voxel is updated based on the
        aggregated predictions.
    """
    xy_coords = vtc.project_coordinates(torch_voxels, intrinsics, extrinsics, give_prod=False)
    # permute x and y coordinates
    xy_coords[:, 2, :] = xy_coords[:, 0, :]
    xy_coords[:, 0, :] = xy_coords[:, 1, :]
    xy_coords[:, 1, :] = xy_coords[:, 2, :]

    coords = vtc.correct_coords_outside(xy_coords, Sx, Sy, xinit, yinit,
                                        -1)  # correct the coordinates that project outside
    xy_full_flat = vtc.flatten_coordinates(coords, the_shape)
    assign_preds = preds_flat[xy_full_flat].reshape(pred_pad.shape[0],
                                                    xy_full_flat.shape[0] // pred_pad.shape[0], preds_flat.shape[-1])
    del xy_full_flat

    assign_preds = torch.sum(assign_preds, dim=0)
    assign_preds[:, 0] *= 0.8
    torch_voxels[:, 3] = torch.argmax(assign_preds, dim=1)
    return torch_voxels


class Classification(torch.nn.Module):
    """Neural network module for classification tasks.

    It initializes a sequence of layers, consisting of a single linear layer, and
    defines a forward pass through the network. The input dimensionality must
    match the dimension provided during initialization.

    Attributes
    ----------
    layer : torch.nn.Sequential
        The sequential container holding the layers of the neural network.
    """

    def __init__(self, D_in, D_out):
        super(Classification, self).__init__()
        lin = torch.nn.Linear(D_in, D_in)
        # lin.weight.data = torch.eye(D_in, requires_grad = True).to(device)
        # lin.bias.data.fill_(0)
        self.layer = torch.nn.Sequential(lin)

    def forward(self, x):
        """Implements the forward propagation for a neural network layer.

        This method applies the computations of a neural network layer
        using the input tensor `x` and returns the output tensor `out`.
        The layer computation logic should be defined in `self.layer`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the layer. The specific shape and dtype of
            the tensor depend on the particular application and model architecture.

        Returns
        -------
        out : torch.Tensor
            Output tensor after applying the layer's computation logic
            on the input tensor `x`. The shape and dtype of the output tensor
            depend on the specifics of `self.layer` processing the input.
        """
        out = self.layer(x)
        return out


class ResNetUNet_3D(nn.Module):
    """
    ResNetUNet_3D class integrates a 3D U-Net architecture with ResNet101 as an encoder
    for image segmentation tasks. The network performs feature extraction using
    pretrained ResNet101, followed by upsampling and concatenation during the decoding
    process to reconstruct the full-resolution segmented image. It also includes
    post-processing to adjust predictions based on coordinate data.

    This class is suitable for segmentation tasks which require the integration of
    coordinate-based processing and ensures flexibility in learning-based applications.

    Attributes
    ----------
    n_class : int
        Number of classes for segmentation or classification.
    base_model : torchvision.models.ResNet
        Pretrained ResNet101 model used as the encoder.
    base_layers : list
        List of layers extracted from ResNet101 for feature extraction.
    layer0 : torch.nn.Sequential
        Initial convolutional and batch normalization layers from ResNet101.
    layer0_1x1 : torch.nn.Sequential
        Convolutional layer for reducing depth after layer0.
    layer1 : torch.nn.Sequential
        Second convolutional block from ResNet101, including down-sampling.
    layer1_1x1 : torch.nn.Sequential
        Convolutional layer for reducing depth after layer1.
    layer2 : torch.nn.Sequential
        Third convolutional block, used for deeper feature extraction.
    layer2_1x1 : torch.nn.Sequential
        Convolutional layer for reducing depth after layer2.
    layer3 : torch.nn.Sequential
        Fourth convolutional block, deeper representation of extracted features.
    layer3_1x1 : torch.nn.Sequential
        Convolutional layer for reducing depth after layer3.
    layer4 : torch.nn.Sequential
        Fifth and final convolutional block from ResNet101.
    layer4_1x1 : torch.nn.Sequential
        Convolutional layer for reducing depth after layer4.
    upsample : torch.nn.Upsample
        Upsampling layer for increasing spatial dimensions during decoding.
    conv_up3 : torch.nn.Sequential
        Convolutional decoder layer combining layer3 and upsampled layer4.
    conv_up2 : torch.nn.Sequential
        Convolutional decoder layer combining layer2 and upsampled intermediate features.
    conv_up1 : torch.nn.Sequential
        Convolutional decoder layer combining layer1 and upsampled intermediate features.
    conv_up0 : torch.nn.Sequential
        Convolutional decoder layer combining layer0 and upsampled intermediate features.
    conv_original_size0 : torch.nn.Sequential
        Initial convolutional layer for input preprocessing.
    conv_original_size1 : torch.nn.Sequential
        Convolutional layer to process features from conv_original_size0.
    conv_original_size2 : torch.nn.Sequential
        Convolutional decoder layer to restore original feature dimensions.
    conv_last : torch.nn.Conv2d
        Final convolutional layer outputting class scores for segmentation.
    class_layer : torch.nn.Sequential
        Post-processing layer to normalize and adjust class predictions.
    coord_file_loc : str
        Path to the coordinate file used for prediction adjustments.
    """

    def __init__(self, n_class, coord_file_loc):
        super().__init__()

        # Use ResNet101 as the encoder with the pretrained weights
        self.n_class = n_class
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

        lin = torch.nn.Linear(n_class + 1, n_class)
        lin.weight.data.fill_(0)
        lin.weight.data.fill_diagonal_(1)
        lin.bias.data.fill_(0)
        self.class_layer = nn.Sequential(lin,
                                         nn.ReLU(inplace=True))

        self.coord_file_loc = coord_file_loc
        # self.xinit = xinit
        # self.yinit = yinit
        # self.Sx = Sx
        # self.Sy = Sy

    def forward(self, input):
        """Perform the forward pass through the model for generating predictions based on input.

        This method takes an input tensor and passes it through multiple layers of the model,
        including convolutional operations, upsample layers, concatenations, and final postprocessing
        to generate spatially adjusted predictions for the given input. The output includes
        intermediate processed features and a final prediction tensor. The predictions are adjusted
        using predefined coordinates and reshaped appropriately to form the output.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor representing the data for the forward pass, typically provided in a
            format compatible with the model's expected input dimensions.

        Returns
        -------
        list
            A list containing two output tensors:
            - The final processed feature tensor after convolutional and upsample operations.
            - The adjusted prediction tensor, computed by performing additional operations and
              transformations on the predicted outputs.
        """
        x_original = self.conv_original_size0(input)
        N_frames = x_original.shape[0]

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
        x = self.conv_last(x)

        xy_full_flat = torch.load(self.coord_file_loc + '/coords.pt').to(device)

        # haut = torch.empty(N_red, label_num, (xinit-Sx)//2, yinit, requires_grad = True).to(device)
        # bas = torch.empty(N_red, label_num, (xinit-Sx)//2, yinit, requires_grad = True).to(device)
        # gauche = torch.empty(N_red, label_num, Sx, (yinit - Sy)//2, requires_grad = True).to(device)
        # droite = torch.empty(N_red, label_num, Sx, (yinit - Sy)//2, requires_grad = True).to(device)

        # pred_pad = torch.cat((gauche, x, droite), dim = 3)
        # pred_pad = torch.cat((haut, pred_pad, bas), dim = 2)
        # pred_pad = Variable(pred_pad, requires_grad = True)
        # pred_pad[:,:,(xinit-Sx)//2:(xinit+Sx)//2,
        # (yinit-Sy)//2:(yinit+Sy)//2] = x #To fit the camera parameters

        # pred_pad = pred_pad.permute(0,2,3,1)
        # print(preds.shape)
        pred_pad = F.sigmoid(torch.flip(x, dims=[0])).permute(0, 2, 3, 1)
        pred_pad = vtc.adjust_predictions(pred_pad)
        # print(preds.shape)
        pred_pad = pred_pad[xy_full_flat].reshape(N_frames,
                                                  xy_full_flat.shape[0] // N_frames, pred_pad.shape[-1])
        # print(preds.shape)
        # preds[:,:,6] = 0
        # print(preds.shape, n_class)

        pred_pad = self.class_layer(pred_pad)
        # pred_pad = pred_pad.clamp(min=1e-8)
        # pred_pad = torch.log(pred_pad)
        # pred_pad  = torch.sum(pred_pad, dim = 0)
        pred_pad = torch.prod(pred_pad, dim=0)
        # print(pred_pad.shape)
        # print(preds.shape)
        # print(torch.max(preds, dim = 0))
        # print(preds.shape)
        del xy_full_flat

        return [x, pred_pad]