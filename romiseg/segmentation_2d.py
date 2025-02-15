#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation2D Module
======================

This module is designed for segmenting 2D image datasets.
It provides tools for loading image datasets, preprocessing images via transformations, and
modifying image channels to make the data suitable for training or inference pipelines.

Key Features
------------
- `Dataset_im_id`: A custom dataset class for managing image file paths and their associated transformations.
  - Supports indexing for retrieving individual preprocessed images and their unique IDs.
  - Includes functionality to crop, resize, and normalize images using a supplied transformation pipeline.
- Helper functionality for segmenting images via the `segmentation` function, aimed at automating segmentation tasks.
- Logging capabilities for debugging outputs and monitoring progress during advanced operations.


Created on Mon Jul  8 16:17:15 2019
@author: alienor
"""
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from plantdb import io
from romiseg.utils.train_from_dataset import ResizeCrop
from romiseg.utils.train_from_dataset import ResizeFit
from romiseg.utils.train_from_dataset import evaluate
from romiseg.utils.train_from_dataset import model_from_fileset

logger = logging.getLogger('romiscan')


class Dataset_im_id(Dataset):
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


def segmentation(Sx, Sy, images_fileset, model_file, resize=False):
    """Segments a list of images using a pretrained deep learning model.

    This function processes images with resizing, normalization, and forwards them through
    a model to generate output segmentations.

    Parameters
    ----------
    Sx : int
        A target width for resizing images.
    Sy : int
        A target height for resizing images.
    images_fileset : list
        A list of paths to input image files that will be segmented.
    model_file : Any
        A file containing the pretrained model and related metadata.
    resize : bool, optional
        Whether to resize the output to the original image dimensions.
        Defaults to False.

    Returns
    -------
    torch.Tensor
        The segmented output with dimensions adjusted for padding.
    list
        The list of image IDs corresponding to the order of the input files.
    """
    # Determine if a GPU is available; use CPU otherwise
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using {device} for images segmentation.')

    # Set up the preprocessing pipeline: resizing, tensor conversion, and normalization
    trans = transforms.Compose([
        ResizeCrop((Sx, Sy)),  # Custom resizing for input dimensions
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                             std=[0.229, 0.224, 0.225])  # Normalize using ImageNet std deviation
    ])

    # Create a dataset object with image file paths and transformations
    image_set = Dataset_im_id(images_fileset, transform=trans)

    # Create a DataLoader for batch processing (batch_size=1 for handling individual images)
    batch_size = 1
    loader = DataLoader(image_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Save folder
    # directory_weights = appdirs.user_cache_dir()

    # directory_weights = '/home/alienor/Documents/database/WEIGHTS'
    # if directory_weights == 'complete here':
    #   directory_weights = filedialog.askdirectory(initialdir="/home/", title='create folder to save fine-tuning weights')
    #   create_folder_if(directory_weights)

    # Load the pretrained segmentation model and associated label names
    logger.debug(f"Model name: {model_file.get_metadata('model_id')}")
    model_segmentation, label_names = model_from_fileset(model_file)

    # Move the model to the selected device (GPU/CPU)
    model_segmentation = model_segmentation.to(device)

    # Read the first image to get its original size (used for padding operations)
    s = io.read_image(images_fileset[0]).shape  # Shape contains [channels, height, width]
    xinit = s[0]  # Original height of the image
    yinit = s[1]  # Original width of the image
    N_cam = len(images_fileset)  # Number of images in the input fileset

    with torch.no_grad():  # Disable gradient computation for inference
        pred_tot = []  # List to store predictions/segmentations
        id_list = []  # List to store IDs of processed images
        count = 0  # Counter to track the number of processed images
        logger.debug("Starting image segmentation using the CNN.")
        # Placeholder RGB image to determine padding needed for resizing
        im = Image.new("RGB", (Sx, Sy))
        new_size, padding = ResizeFit((xinit, yinit)).padding(im)  # Calculate new size and padding
        # Process each image in the dataloader
        for inputs, id_im in tqdm(loader):
            # Move input image to the same device as the model
            inputs = inputs.to(device)
            # Perform inference using the segmentation model
            outputs = evaluate(inputs, model_segmentation)
            # Resize outputs to include padding using bilinear interpolation
            outputs = F.interpolate(outputs, new_size, mode='bilinear')
            # Store the model's outputs and corresponding image ID
            pred_tot.append(outputs)
            id_list.append(id_im)
            count += 1

    # Combine all predictions into a single tensor
    pred_tot = torch.cat(pred_tot, dim=0)
    # Initialize an empty tensor to contain predictions with padding reversed
    pred_pad = torch.zeros((N_cam, len(label_names), xinit, yinit))
    # Reverse padding applied earlier to match the original image dimensions
    pred_pad[:, :, padding[0]:pred_pad.size(2) - padding[2], padding[1]:pred_pad.size(3) - padding[3]] = pred_tot

    #        if resize:
    #                pass
    #        else:
    #
    #            pred_pad = torch.zeros((N_cam, len(label_names), xinit, yinit)) #reverse the crop in order to match the colmap parameters
    #            pred_pad[:,:,(xinit-Sx)//2:(xinit+Sx)//2,(yinit-Sy)//2:(yinit+Sy)//2] = pred_tot #To fit the camera parameters

    return pred_pad, id_list


def fileset_segmentation(Sx, Sy, images_fileset, model_file):
    """Segment images within a provided dataset using a pre-trained model.

    This function processes a set of input images, applies pre-processing transformations,
    performs segmentation using a pre-trained deep learning model, and adjusts the output
    segmentations to match the original image dimensions. It handles GPU acceleration if
    available, falling back to CPU otherwise.

    Parameters
    ----------
    Sx : int
        The target width for resizing the input images.
    Sy : int
        The target height for resizing the input images.
    images_fileset : list[str or Path]
        A list containing file paths to the input images to be segmented.
    model_file : plantdb.FSDB.File
        A `File` object containing the pre-trained segmentation model and its associated metadata.

    Returns
    -------
    pred_images : list[torch.Tensor]
        A list of tensors where each tensor represents the segmentation result for one input image.
        Dimensions of each tensor match the original input image dimensions.
    id_list : list
        A list of IDs corresponding to the processed input images.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from PIL import Image
    >>> from skimage.morphology import binary_dilation, disk
    >>> from romiseg.segmentation_2d import fileset_segmentation
    >>> from plantdb.test_database import test_database
    >>> db = test_database(with_models=True)
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> image_fs = [im_f.path() for im_f in scan.get_fileset('images').get_files(query={"channel": "rgb"})]
    >>> model_file = db.get_scan('models').get_fileset('models').get_file('Resnet_896_896_epoch50')
    >>> label_names = model_file.get_metadata('label_names')
    >>> print(label_names)
    ['background', 'flower', 'fruit', 'leaf', 'pedicel', 'stem']
    >>> pred_pads, im_ids = fileset_segmentation(896, 896, image_fs, model_file)
    >>> # Export one predicted label in one image for visualization
    >>> img_idx = 0  # index of the image to export
    >>> label_idx = 1  # index of the label to export
    >>> im = pred_pads[img_idx][label_idx, :, :].cpu().numpy()
    >>> im = im > 0.01  # threshold the segmentation mask
    >>> im = binary_dilation(im, footprint=disk(1))  # dilate the segmentation mask
    >>> im = (im * 255).astype(np.uint8)  # convert the segmentation mask to an image
    >>> im = 255 - im if label_names[label_idx] == 'background' else im  # invert the background color
    >>> img = Image.fromarray(im)  # create a PIL image from the segmentation mask
    >>> img.save(f'predicted_{label_names[label_idx]}.png')  # save the segmentation mask as an image
    >>> db.disconnect()
    """
    # Determine if a GPU is available; use CPU otherwise
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using {device} for images segmentation.')

    # Set up the preprocessing pipeline: resizing, tensor conversion, and normalization
    trans = transforms.Compose([
        ResizeCrop((Sx, Sy)),  # Custom resizing for input dimensions
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                             std=[0.229, 0.224, 0.225])  # Normalize using ImageNet std deviation
    ])

    # Create a dataset object with image file paths and transformations
    image_set = Dataset_im_id(images_fileset, transform=trans)

    # Create a DataLoader for batch processing (batch_size=1 for handling individual images)
    batch_size = 1
    loader = DataLoader(image_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load the pretrained segmentation model and associated label names
    logger.debug(f"Model name: {model_file.get_metadata('model_id')}")
    model_segmentation, label_names = model_from_fileset(model_file)

    # Move the model to the selected device (GPU/CPU)
    model_segmentation = model_segmentation.to(device)

    # Read the first image to get its original size (used for padding operations)
    s = io.read_image(images_fileset[0]).shape  # Shape contains [channels, height, width]
    xinit = s[0]  # Original height of the image
    yinit = s[1]  # Original width of the image
    pred_images = []  # List to store predicted images, label ordered
    # Process each image in the dataloader
    for inputs, _ in tqdm(loader):
        # Placeholder RGB image to determine padding needed for resizing
        im = Image.new("RGB", (Sx, Sy))
        new_size, padding = ResizeFit((xinit, yinit)).padding(im)  # Calculate new size and padding
        # Move input image to the same device as the model
        inputs = inputs.to(device)
        with torch.no_grad():  # Disable gradient computation for inference
            # Perform inference using the segmentation model
            outputs = evaluate(inputs, model_segmentation)
            # Resize outputs to include padding using bilinear interpolation
            outputs = F.interpolate(outputs, new_size, mode='bilinear')
        # Initialize an empty tensor to contain predictions with padding reversed
        pred_pad = torch.zeros((len(label_names), xinit, yinit))
        # Reverse padding applied earlier to match the original image dimensions
        pred_pad[:, padding[0]:pred_pad.size(1) - padding[2], padding[1]:pred_pad.size(2) - padding[3]] = outputs.squeeze(0)
        pred_images.append(pred_pad)

    return pred_images

def file_segmentation(Sx, Sy, image_path, model_segmentation, label_names, device, resize=False,):
    """Segments a single image using a pretrained deep learning model.

    Parameters
    ----------
    Sx : int
        A target width for resizing the image.
    Sy : int
        A target height for resizing the image.
    image_path : str
        The path to the input image file that will be segmented.
    model_segmentation : Any
        A file containing the pretrained model and related metadata.
    label_names : list
        A list of label names corresponding to the model's output.'
    device : torch.device
        A device object representing the target device for processing.

    Returns
    -------
    torch.Tensor
        The segmented output with dimensions adjusted for padding.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from PIL import Image
    >>> from skimage.morphology import binary_dilation, disk
    >>> from romiseg.segmentation_2d import file_segmentation
    >>> from romiseg.utils.train_from_dataset import model_from_fileset
    >>> from plantdb.test_database import test_database
    >>> db = test_database(with_models=True)
    >>> db.connect()
    >>> scan = db.get_scan("real_plant_analyzed")
    >>> image_fs = scan.get_fileset('images')
    >>> images_path = [image.path() for image in image_fs.get_files(query={"channel": "rgb"})]
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> model_file = db.get_scan('models').get_fileset('models').get_file('Resnet_896_896_epoch50')
    >>> print(model_file.metadata)
    {'label_names': ['background', 'flower', 'fruit', 'leaf', 'pedicel', 'stem'], 'model_id': 'Resnet_896_896_epoch50'}
    >>> model_segmentation, label_names = model_from_fileset(model_file)
    >>> model_segmentation = model_segmentation.to(device)
    >>> pred_pad = file_segmentation(896, 896, images_path[0], model_segmentation, label_names, device)
    >>> # Export one predicted label in the given image for visualization
    >>> label_idx = 1  # index of the label to export
    >>> im = pred_pad[label_idx, :, :].cpu().numpy()
    >>> im = im > 0.01  # threshold the segmentation mask
    >>> im = binary_dilation(im, footprint=disk(1))  # dilate the segmentation mask
    >>> im = (im * 255).astype(np.uint8)  # convert the segmentation mask to an image
    >>> im = 255 - im if label_names[label_idx] == 'background' else im  # invert the background color
    >>> img = Image.fromarray(im)  # create a PIL image from the segmentation mask
    >>> img.save(f'predicted_{label_names[label_idx]}.png')  # save the segmentation mask as an image
    >>> db.disconnect()
    """
    # Set up the preprocessing pipeline: resizing, tensor conversion, and normalization
    trans = transforms.Compose([
        ResizeCrop((Sx, Sy)),  # Custom resizing for input dimensions
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                             std=[0.229, 0.224, 0.225])  # Normalize using ImageNet std deviation
    ])

    # Load and preprocess the image
    image = Image.open(image_path)
    original_size = image.size  # (width, height)
    xinit, yinit = original_size[1], original_size[0]  # Original height and width

    # Placeholder RGB image to determine padding needed for resizing
    temp_image = Image.new("RGB", (Sx, Sy))
    new_size, padding = ResizeFit((xinit, yinit)).padding(temp_image)  # Calculate new size and padding

    # Transform the input image
    input_image = trans(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():  # Disable gradient computation for inference
        logger.debug("Starting single image segmentation using the CNN.")
        # Perform inference using the segmentation model
        outputs = evaluate(input_image, model_segmentation)
        # Resize outputs to include padding using bilinear interpolation
        outputs = F.interpolate(outputs, new_size, mode='bilinear')

    # Initialize an empty tensor to contain predictions with padding reversed
    pred_pad = torch.zeros((len(label_names), xinit, yinit))

    # Reverse padding applied earlier to match the original image dimensions
    pred_pad[:, padding[0]:pred_pad.size(1) - padding[2], padding[1]:pred_pad.size(2) - padding[3]] = outputs.squeeze(0)

    return pred_pad
