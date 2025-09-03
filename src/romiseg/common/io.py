#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# I/O Utilities.

This module contains utilities for file I/O operations.
"""

import os
import requests
import torch
import numpy as np
from plantdb.commons import io

# Current working directory
cwd = os.getcwd()


def create_folder_if(directory):
    """
    Creates a directory if it doesn't exist.
    
    Parameters
    ----------
    directory : str
        Path to the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def catch_file(direc=cwd):
    """
    Opens a dialog window to select a file.
    
    Parameters
    ----------
    direc : str, optional
        Directory to open. Default is current directory.
        
    Returns
    -------
    str
        Path to the selected file.
    """
    from PyQt5 import QtWidgets
    fname = QtWidgets.QaFileDialog.getOpenFileName(None, directory=direc, caption="Select a video file...",
                                                  filter="All files (*)")
    return fname[0]


def download_file(url, target_dir):
    """
    Downloads a file from a URL to a target directory.
    
    Parameters
    ----------
    url : str
        URL of the file to download.
    target_dir : str
        Directory to save the file to.
        
    Returns
    -------
    str
        Name of the downloaded file.
    """
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(target_dir + '/' + local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename


def save_and_load_model(weights_folder, model_segmentation_name):
    """
    Saves and loads a model.
    
    Parameters
    ----------
    weights_folder : str
        Directory to save the model to.
    model_segmentation_name : str
        Name of the model file.
        
    Returns
    -------
    torch.nn.Module
        The loaded model.
    """
    # if not already saved, download from database
    if model_segmentation_name not in os.listdir(weights_folder):
        url = 'http://db.romi-project.eu/models/' + model_segmentation_name
        download_file(url, weights_folder)

    model_segmentation = torch.load(weights_folder + '/' + model_segmentation_name)
    try:
        model_segmentation = model_segmentation[0]
    except:
        model_segmentation = model_segmentation

    try:
        model_segmentation = model_segmentation.module
    except:
        model_segmentation = model_segmentation

    return model_segmentation


def model_from_fileset(model_file):
    """
    Loads a machine learning model and its associated label names from a specified file.

    Parameters
    ----------
    model_file : plantd.FSDB.File
        A `File` object containing the serialized PyTorch model as well as its associated metadata.

    Returns
    -------
    torch.nn.Module
        The core part of the loaded PyTorch model after processing.
    numpy.ndarray
        A sorted array of label names associated with the model.
    """
    model_segmentation = io.read_torch(model_file)
    label_names = model_file.get_metadata('label_names')

    if not isinstance(model_segmentation, torch.nn.Module):
        from romiseg.models.unet import ResNetUNet
        model_segmentation = ResNetUNet(len(label_names))
        model_segmentation.load_state_dict(io.read_torch(model_file))
        return model_segmentation, np.sort(label_names)

    try:
        model_segmentation = model_segmentation[0]
    except:
        model_segmentation = model_segmentation

    try:
        model_segmentation = model_segmentation.module
    except:
        model_segmentation = model_segmentation

    return model_segmentation, np.sort(label_names)