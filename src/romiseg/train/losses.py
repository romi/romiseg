#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loss functions for training segmentation models.

This module contains loss functions used for training segmentation models.
"""

import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.):
    """
    Calculates the Dice loss, a common loss function used for image segmentation tasks, which maximizes the overlap
    between predicted and target binary masks. The Dice loss can handle imbalanced datasets by focusing on the
    overlap rather than absolute pixel accuracy.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted binary mask, typically the model's output, expected to have the shape
        (batch_size, channels, height, width).

    target : torch.Tensor
        Ground truth binary mask, expected to have the same shape as `pred`.

    smooth : float, optional
        A smoothing factor to avoid division by zero or undefined values, default is 1.

    Returns
    -------
    torch.Tensor
        The scalar value representing the mean Dice loss for all the batches in the input.
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    """
    Computes a weighted loss combining Binary Cross Entropy (BCE) and Dice loss.

    This function calculates the BCE loss and the Dice loss, incorporates a
    weighting factor between the two, and tracks the metrics for each category.
    The loss is computed per batch, and the metrics dictionary is updated with
    the cumulative loss values.

    Parameters
    ----------
    pred : torch.Tensor
        The predicted logits from the model. Should have the same shape as
        `target`.
    target : torch.Tensor
        The ground truth binary labels. Should have the same shape as `pred`.
    metrics : dict
        A dictionary to store cumulative loss values for BCE, Dice, and overall
        loss. This dictionary is updated in-place.
    bce_weight : float, optional
        The weight to balance BCE loss and Dice loss. Defaults to 0.5, with equal
        emphasis on both losses.

    Returns
    -------
    torch.Tensor
        The weighted loss value for the batch, combining BCE and Dice losses.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss