#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Dataset Utilities.

This module contains utilities for working with datasets.
"""

import copy
import numpy as np
from plantdb.commons import fsdb


def init_set(mode, path):
    """
    Initialize a dataset from a path.
    
    Parameters
    ----------
    mode : str
        Mode for initialization (currently unused).
    path : str
        Path to the dataset.
        
    Returns
    -------
    list
        List of shots.
    numpy.ndarray
        Sorted array of channels.
    """
    db = fsdb.FSDB(path)  # Initialize database connection
    db.connect()  # Connect to the database
    scans = db.get_scans()  # Retrieve all scans from the database
    shots = []  # List to hold shot information

    for s in scans:
        f = s.get_fileset('images')  # Get fileset named 'images' from scan
        list_files = f.get_files(query={'channel': 'rgb'})
        # Build a list of shots with scan ID and shot ID metadata
        shots += [{"scan": s.id, "shot_id": list_files[i].metadata['shot_id']} for i in range(len(list_files))]

    channels = f.get_metadata('channels')  # Retrieve channel metadata from fileset
    channels = copy.copy(channels)  # Create a copy of the channels list
    channels.remove('rgb')  # Remove 'rgb' channel from the list

    db.disconnect()  # Disconnect from the database

    return shots, np.sort(channels)