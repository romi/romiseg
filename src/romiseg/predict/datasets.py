#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset classes for prediction with segmentation models.

This module contains dataset classes used for making predictions with segmentation models.
"""

import logging
from pathlib import Path

from PIL import Image
from plantdb.commons import io
from torch.utils.data import Dataset

logger = logging.getLogger('romiseg')


class DatasetImId(Dataset):
    """Represents a dataset that includes image file paths and their corresponding transformations.

    This class provides methods to access image data from file paths, apply
    transformations, and retrieve elements by index. It is typically used in
    image processing tasks that require preparing datasets for training or
    inference.

    Attributes
    ----------
    image_paths : list
        A list of file paths to the image dataset.
    transforms : callable
        A callable transformation function or a pipeline of transformations to
        be applied to the images.
    """

    def __init__(self, image_paths, transform):
        """Initializes a new instance.

        Parameters
        ----------
        image_paths : list of str
            A list of strings representing the paths of image files.
        transform : callable or None
            A callable object (e.g., a function) that defines the set of transformations
            to be applied to the images. If None, no transformations are applied.
        """
        self.image_paths = image_paths
        self.transforms = transform

    def __getitem__(self, index):
        """Retrieve an image and its corresponding ID from the dataset through indexing.

        Parameters
        ----------
        index : int
            The index of the requested image in the dataset.

        Returns
        -------
        torch.Tensor
            The transformed image with only RGB channels.
        str
            The image ID, providing a unique identifier for the image.
        """
        if not self.image_paths:
            raise IndexError(f"Empty dataset. Cannot access index '{index}'.")
        # Extract the file path of the image from the dataset using the provided index.
        db_file = self.image_paths[index]
        # Read the image from the file and convert it to a Pillow Image object.
        # The slicing [:, :, :3] ensures only the first three channels (R, G, B) are used.
        image = Image.fromarray(io.read_image(db_file)[:, :, :3])

        # Apply the transformations (e.g., cropping, resizing, normalization) defined for this dataset.
        t_image = self.transforms(image)
        # Select the first three channels (R, G, B) from the transformed image.
        # This ensures that any additional channels (e.g., alpha) are removed.
        t_image = t_image[0:3, :, :]

        # Debugging line for checking the maximum pixel value in the transformed image:
        logger.debug(f"Max pixel value in transformed image: {t_image.max()}")
        return t_image, Path(db_file).name

    def __len__(self):
        """Returns the number of images present in the dataset.

        Returns
        -------
        int
            The number of images present in the dataset.
        """
        return len(self.image_paths)
