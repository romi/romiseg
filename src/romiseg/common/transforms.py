#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image transformation utilities for the romiseg package.

This module contains utilities for transforming images.
"""

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.autograd import Variable


def gaussian(ins, is_training, mean, stddev, dyn=1):
    """
    Adds Gaussian noise to a tensor.
    
    Parameters
    ----------
    ins : torch.Tensor
        Input tensor.
    is_training : bool
        Whether the model is in training mode.
    mean : float
        Mean of the Gaussian noise.
    stddev : float
        Standard deviation of the Gaussian noise.
    dyn : float, optional
        Dynamic range of the output. Default is 1.
        
    Returns
    -------
    torch.Tensor
        The tensor with added Gaussian noise.
    """
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return torch.clamp(ins + noise, -dyn, dyn)
    return torch.clamp(ins, -dyn, dyn)


class ResizeFit(object):
    """
    Resize an image to fit within a given size while preserving aspect ratio.
    
    Parameters
    ----------
    size : tuple
        Target size (width, height).
    interpolation : int, optional
        Interpolation method. Default is PIL.Image.BILINEAR.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def padding(self, img):
        aspect_ratio = self.size[0] / self.size[1]
        old_aspect_ratio = img.size[0] / img.size[1]

        if aspect_ratio < old_aspect_ratio:
            new_size = (self.size[0], int(1 / old_aspect_ratio * self.size[0]))
        else:
            new_size = (int(old_aspect_ratio * self.size[1]), self.size[1])

        diff = [self.size[i] - new_size[i] for i in range(2)]
        padding = diff[0] // 2, diff[1] // 2, (diff[0] + 1) // 2, (diff[1] + 1) // 2
        return new_size, padding

    def __call__(self, img):
        from PIL import ImageOps
        new_size, padding = self.padding(img)
        new_img = img.resize(new_size, resample=self.interpolation)
        new_img = ImageOps.expand(new_img, padding)
        return new_img


class ResizeCrop(object):
    """
    Resize an image and then crop it to a given size.
    
    Parameters
    ----------
    size : tuple
        Target size (width, height).
    interpolation : int, optional
        Interpolation method. Default is PIL.Image.BILINEAR.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def padding(self, img):
        aspect_ratio = self.size[0] / self.size[1]
        old_aspect_ratio = img.size[0] / img.size[1]

        if aspect_ratio > old_aspect_ratio:
            new_size = (self.size[0], int(1 / old_aspect_ratio * self.size[0]))
        else:
            new_size = (int(old_aspect_ratio * self.size[1]), self.size[1])

        diff = [- self.size[i] + new_size[i] for i in range(2)]
        padding = diff[0] // 2, diff[1] // 2, (diff[0] + 1) // 2, (diff[1] + 1) // 2

        return new_size, padding

    def __call__(self, img):
        from PIL import ImageOps
        new_size, padding = self.padding(img)
        new_img = img.resize(new_size, resample=self.interpolation)
        new_img = ImageOps.crop(new_img, padding)
        return new_img


class MyRotationTransform:
    """
    Rotate an image by a given angle.
    
    Parameters
    ----------
    angle : float
        Angle to rotate the image by.
    fill : list[float], optional
        Fill color for the rotated image.
    """

    def __init__(self, angle, fill):
        self.angle = angle
        self.fill = fill

    def __call__(self, x):
        return TF.rotate(x, self.angle, fill=self.fill)
