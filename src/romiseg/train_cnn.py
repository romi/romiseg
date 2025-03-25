#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CNN Training CLI

This script provides tools for configuring, training, and managing convolutional neural network (CNN) models.

Key Features:
- **Configuration Parsing**: Utilities (`parser`, `parse_config`) for handling experiment configurations, ensuring flexible and streamlined setup.
- **Model Training**: A `cnn_train` function for training neural networks with support for custom datasets, batch sizes, epochs, and more.
- **Experiment Management**: Includes features for managing and saving trained models and tracking metadata to ensure reproducibility.

Created on Thu Nov 21 09:18:24 2019

@author: alienor
"""

import argparse
import os

import toml
import torch
import torch.optim as optim
from plantdb.commons import fsdb
from plantdb.commons import io
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from romiseg.utils import segmentation_model
from romiseg.utils.train_from_dataset import Dataset_im_label
from romiseg.utils.train_from_dataset import init_set
from romiseg.utils.train_from_dataset import plot_dataset
from romiseg.utils.train_from_dataset import train_model

# from torchvision import models


default_config = "/home/alienor/Documents/scanner-meta-repository/Scan3D/default/segmentation2d_arabidopsis.toml"


def parser():
    """Parses command-line arguments.

    Returns
    -------
    argparse.ArgumentParser
        A configured argument parser with an option for specifying the configuration directory.
    """
    parser = argparse.ArgumentParser(description='Train a CNN model on a dataset.')

    parser.add_argument('--config', dest='config', type=str, default=default_config,
                        help='config dir, default: %s' % default_config)

    return parser


def parse_config(config):
    """Parses configuration file for a 2D segmentation task and retrieves relevant parameters.

    Parameters
    ----------
    config : str
        Path to the configuration TOML file to be parsed.

    Returns
    -------
    str
        Path to the dataset directory.
    str
        Path where the weights will be saved.
    str
        Path for TensorBoard log directory specific to 2D segmentation.
    int
        Batch size for training.
    int
        Number of training epochs.
    int
        Size of the input in the X dimension for the model.
    int
        Size of the input in the Y dimension for the model.
    str
        Name of the segmentation model being used.
    float
        Learning rate for the optimization process.
    """
    param_pipe = toml.load(config)
    direc = param_pipe['TrainingDirectory']
    path = direc['path']
    directory_weights = path + direc['directory_weights']
    tsboard = path + direc['tsboard'] + '/2D_segmentation'
    directory_dataset = path + direc['directory_dataset']
    param2 = param_pipe['Segmentation2D']
    model_name = param2["model_name"]
    # label_names = param2['labels'].split(',')
    Sx = param2['Sx']
    Sy = param2['Sy']
    epochs = param2['epochs']
    batch_size = param2['batch']
    learning_rate = param2['learning_rate']
    return directory_dataset, directory_weights, tsboard, batch_size, epochs, Sx, Sy, model_name, learning_rate


def cnn_train(f_weights, directory_dataset, label_names, tsboard, batch_size, epochs,
              model, Sx, Sy, learning_rate, data_augmentation=True):
    """Train a Convolutional Neural Network (CNN) on a given dataset.

    This function allows for the customization of training parameters such as batch size, epochs, and optimizer,
    as well as dataset-specific properties like image dimensions and augmentation.
    The training progress is logged using TensorBoard, and the trained model can optionally be saved.

    Parameters
    ----------
    f_weights : str
        Path to save the model weights after training.
    directory_dataset : str
        Path to the dataset directory containing `train`, `val`, and `test` subdirectories.
    label_names : list of str
        List of labels corresponding to the dataset classes.
    tsboard : str
        Directory where TensorBoard logs will be saved.
    batch_size : int
        Batch size for training and validation data loaders.
    epochs : int
        Number of epochs to train the model.
    model : torch.nn.Module
        The PyTorch model to train.
    Sx : int
        Width of the input images to the model.
    Sy : int
        Height of the input images to the model.
    load_model : bool, optional
        Whether to load a pre-trained model at the start (default is False).
    showit : bool, optional
        Whether to visualize the training dataset images (default is False).
    data_augmentation : bool, optional
        Whether to apply data augmentation on the training dataset (default is True).

    Returns
    -------
    torch.nn.Module
        The trained PyTorch model.
    """
    # Training board
    writer = SummaryWriter(tsboard)

    # Load images and ground truth
    path_val = directory_dataset + '/val/'
    path_train = directory_dataset + '/train/'
    path_test = directory_dataset + '/test/'
    image_train, channels = init_set('', path_train)
    image_val, channels = init_set('', path_val)
    image_test, channels = init_set('', path_test)

    train_dataset = Dataset_im_label(image_train, channels, size=(Sx, Sy), path=path_train,
                                     data_augmentation=data_augmentation)
    val_dataset = Dataset_im_label(image_val, channels, size=(Sx, Sy), path=path_val,
                                   data_augmentation=data_augmentation)
    test_dataset = Dataset_im_label(image_test, channels, size=(Sx, Sy), path=path_test,
                                    data_augmentation=data_augmentation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Show input images
    fig = plot_dataset(train_loader, label_names, batch_size, showit=True)  # display training set
    writer.add_figure('Dataset images', fig, 0)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
    }

    # Choice of optimizer, can be changed
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # make learning rate evolve
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    # Run training
    model = train_model(f_weights, dataloaders, model, optimizer_ft, exp_lr_scheduler, writer,
                        num_epochs=epochs, viz=True, label_names=label_names)
    # save model

    return model


def train_and_save_model(directory_dataset, directory_weights, tsboard, batch_size, epochs, Sx, Sy, model_name,
                         learning_rate):
    """Trains a segmentation model using provided dataset and configurations.

    This function performs the following steps:
    1. Initializes the dataset and retrieves available channels.
    2. Checks if a GPU is available, otherwise defaults to CPU.
    3. Initializes the segmentation model with the total number of channels and moves it to the selected device.
    4. Optionally loads pre-trained model weights.
    5. Freezes the base layers (encoder) of the model to prevent their weights from updating during training.
    6. Trains the CNN model using the provided dataset, channels, and training parameters.
    7. Creates & saves the model weights to a file with a unique name.
    8. Attaches metadata to the file for tracking purposes (e.g., model ID and label names).

    Parameters
    ----------
    directory_dataset : str
        Path to the dataset directory, containing the training data.
    directory_weights : str
        Directory path for saving the model weights.
    tsboard : str
        Identifier for TensorBoard logs to track training progress.
    batch_size : int
        Size of data batches used during training.
    epochs : int
        Number of passes through the full training dataset.
    Sx : int
        Horizontal dimension for image input size.
    Sy : int
        Vertical dimension for image input size.
    model_name : str
        Base name for saving the trained model file.
    learning_rate : float
        Learning rate used for updating model weights during training.
    """
    # Define the path to the training dataset
    path_train = f"{directory_dataset}/train/"

    # Initialize the dataset and retrieve available channels
    image_train, channels = init_set('', path_train)

    # Check if a GPU is available, otherwise default to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the segmentation model (ResNetUNet) with the total channels and move it to the selected device
    model = segmentation_model.ResNetUNet(len(channels)).to(device)

    # Optional: Load pre-trained model weights (commented out)
    # model = save_and_load_model(directory_weights, model_segmentation_name)

    # Freeze the base layers (encoder) of the model to prevent their weights from updating during training
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    # Initialize the file system database (FSDB) for model weights management
    db = fsdb.FSDB(directory_weights)

    # Retrieve a scan object (allows metadata tracking) and create it if it doesn't exist
    s = db.get_scan('models', create=True)

    # Create or retrieve a fileset for storing model weights in the scan
    f_weights = s.get_fileset('models', create=True)

    # Train the CNN model using the provided dataset, channels, and training parameters
    model = cnn_train(
        f_weights,  # Fileset for saving model weights
        directory_dataset,  # Dataset directory
        channels,  # Channel information (e.g., input image channels)
        f"{tsboard}_{Sx}_{Sy}{directory_dataset}",  # TensorBoard log identifier
        batch_size,  # Training batch size
        epochs,  # Number of training epochs
        model,  # The initialized model
        Sx,  # Horizontal input dimension
        Sy,  # Vertical input dimension
        learning_rate,  # learning rate
    )

    # Generate a unique model name for saving based on dataset, input size, and epochs
    model_name = f"{model_name}{os.path.split(directory_dataset)[1]}_{Sx}_{Sy}_epoch{epochs}"

    # Create a file for saving the model weights
    file = f_weights.create_file(model_name)

    # Save the model weights to the created file
    io.write_torch(file, model)

    # Attach metadata to the file for tracking purposes (e.g., model ID and label names)
    file.set_metadata({
        'model_id': model_name,  # Unique identifier for the trained model
        'label_names': channels.tolist(),  # List of channel names used in training
        'batch_size': batch_size,  # Training batch size
        'epochs': epochs,  # Number of training epochs
        'Sx': Sx,  # Horizontal input dimension
        'Sy': Sy,  # Vertical input dimension
        'learning_rate': learning_rate,  # Learning rate for the optimizer
    })


def main():
    args = parser().parse_args()
    train_and_save_model(*parse_config(args.config))


if __name__ == '__main__':
    main()
