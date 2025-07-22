#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation functions for segmentation models.

This module contains functions for evaluating segmentation models.
"""

import torch
import torch.nn.functional as F
from collections import defaultdict

from romiseg.train.losses import calc_loss


def evaluate(inputs, model):
    """
    Evaluate a segmentation model on input images.
    
    Parameters
    ----------
    inputs : torch.Tensor
        Input images.
    model : torch.nn.Module
        Segmentation model.
        
    Returns
    -------
    torch.Tensor
        Predicted segmentation masks.
    """
    with torch.no_grad():
        inputs.requires_grad = False
        # Get the first batch - device is determined by inputs tensor
        device = inputs.device
        # Ensure model is on the same device as inputs
        model = model.to(device)

        pred = model(inputs)
        # The loss functions include the sigmoid function.
        # for i in range(pred.shape[1]):
        #    pred[:,1,:,:] = F.sigmoid(pred[:,1,:,:])

        pred = F.sigmoid(pred)

    return pred


def test(inputs, labels, model):
    """
    Test a segmentation model on input images and labels.
    
    Parameters
    ----------
    inputs : torch.Tensor
        Input images.
    labels : torch.Tensor
        Ground truth labels.
    model : torch.nn.Module
        Segmentation model.
        
    Returns
    -------
    tuple
        Tuple containing the predicted segmentation masks and a dictionary of metrics.
    """
    metrics = defaultdict(float)

    with torch.no_grad():
        inputs.requires_grad = False
        # Get the first batch - device is determined by inputs tensor
        device = inputs.device
        # Ensure model is on the same device as inputs
        model = model.to(device)

        pred = model(inputs)
        # The loss functions include the sigmoid function.
        # for i in range(pred.shape[1]):
        #    pred[:,1,:,:] = F.sigmoid(pred[:,1,:,:])
        loss = calc_loss(pred, labels, metrics)

        pred = F.sigmoid(pred)

    return pred, metrics