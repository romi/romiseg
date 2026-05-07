#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# I/O Utilities.

This module contains utilities for file I/O operations.
"""

import os
from pathlib import Path

import requests
import torch

from plantdb.commons import io
from romiseg.cli.convert_models import convert_model_to_state_dict
from romiseg.cli.convert_models import setup_module_alias
from romiseg.log import get_logger

logger = get_logger(__name__)
# Current working directory
cwd = os.getcwd()


def download_file(url: str | Path, target_dir: str | Path) -> Path:
    """
    Downloads a file from a URL to a target directory.

    Parameters
    ----------
    url : str or pathlib.Path
        URL of the file to download.
    target_dir : str or pathlib.Path
        Directory to save the file to.

    Returns
    -------
    pathlib.Path
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
    return Path(local_filename)


def save_and_load_model(weights_folder: str | Path, model_segmentation_name: str, model_labels: list[str]) -> torch.nn.Module:
    """
    Saves and loads a model.
    
    Parameters
    ----------
    weights_folder : str or pathlib.Path
        Directory to save the model to.
    model_segmentation_name : str
        Name of the model file.
    model_labels : list[str]
        The label names associated with the model.

    Returns
    -------
    torch.nn.Module
        The loaded model.
    """
    # if not already saved, download from database
    if model_segmentation_name not in os.listdir(weights_folder):
        url = 'http://db.romi-project.eu/models/' + model_segmentation_name
        download_file(url, weights_folder)

    return model_from_file(weights_folder + '/' + model_segmentation_name, model_labels)


def model_from_file(model_file: str | Path, model_labels: list[str]) -> torch.nn.Module:
    """
    Loads a machine learning model and its associated label names from a specified file.

    Parameters
    ----------
    model_file : str or pathlib.Path
        A path to the serialized PyTorch model.
    model_labels : list[str]
        The label names associated with the model.

    Returns
    -------
    torch.nn.Module
        The core part of the loaded PyTorch model after processing.

    Examples
    --------
    >>> from plantdb.commons.test_database import test_database
    >>> from romiseg.common.io import model_from_file
    >>> # Set up a test database with an old API model with weights_only=True
    >>> db = test_database('real_plant', with_models=True, no_auth=True)
    >>> db.connect()
    >>> # Get the model fileset and model file
    >>> model_name = 'Resnet_896_896_epoch50.pt'
    >>> models_fileset = db.get_scan('models').get_fileset('models')
    >>> models_file = models_fileset.get_file(model_name.split('.')[0])
    >>> # Get the label names from the model metadata
    >>> label_names = models_file.get_metadata('label_names')
    >>> model_segmentation = model_from_file(models_file.path(), label_names)
    >>> assert model_segmentation is not None
    """
    # Ensure the old‑API module alias exists (needed for some old checkpoints)
    setup_module_alias()

    try:
        # Original behavior: try to read the full checkpoint directly
        # PyTorch 2.6 => introduce `weights_only=True` by default for a more secure behavior
        model_segmentation, _ = io.read_torch(model_file, weights_only=True)
    except Exception as e:
        logger.warning(f"Direct loading failed:\n{e}.")
        logger.info("Converting checkpoint to state‑dict...")
        # Convert the original checkpoint to a state‑dict file
        state_dict_path = convert_model_to_state_dict(model_file, overwrite=False)
        if state_dict_path is None or not state_dict_path.exists():
            logger.error("Conversion to state‑dict failed: cannot load model.")
            raise Exception
        logger.info(f"Model loaded from converted state‑dict: {state_dict_path}")
        from romiseg.models.unet import ResNetUNet
        model_segmentation = ResNetUNet(len(model_labels))
        # Load the newly created state‑dict
        model_segmentation.load_state_dict(torch.load(state_dict_path))
    else:
        logger.info("Model loaded directly with `io.read_torch`")

    return model_segmentation
