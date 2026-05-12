#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Evaluation Metrics.

This module contains metrics used for evaluating segmentation models.
"""

import torch


def my_metric(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Computes the average correspondence between predicted and target outputs, excluding
    entries where the label is zero. This function effectively calculates a filtered
    accuracy metric, considering only valid labels and comparing them against the
    corresponding outputs.

    Parameters
    ----------
    outputs : torch.Tensor
        The predicted outputs, typically the result of a model's forward pass.
    labels : torch.Tensor
        The ground truth labels against which predictions are compared, with `0`
        indicating labels to exclude from evaluation.

    Returns
    -------
    torch.Tensor
        A single-element tensor containing the computed average correspondence (mean
        accuracy) for non-zero labels. The value lies in the range [0, 1], where 1
        indicates perfect correspondence.
    """
    inds = labels != 0
    bools = outputs[inds] == labels[inds]

    return torch.mean(bools.float())  # Or thresholded.mean() if you are interested in average across the batch


def print_metrics(metrics, epoch_samples, phase):
    """
    Logs and prints the evaluation metrics for a specific phase of training.

    The function computes the average value of each metric over the total number
    of samples for the given phase (e.g., training or validation) and formats the
    outputs for display.

    Parameters
    ----------
    metrics : dict
        A dictionary containing the cumulative values of different metrics.
    epoch_samples : int
        The total number of samples processed in the current epoch.
    phase : str
        The phase of training or evaluation, typically 'train' or 'val'.

    Returns
    -------
    None
        The function prints the formatted metrics to the console but does not return
        any value.
    """
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))