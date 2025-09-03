#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Wed Nov  6 10:50:33 2019
#
# @author: alienor

"""
# Fine-tuning CLI.

This script handles fine-tuning a model and managing its configuration, making it easier
for users to set up, execute, and modify training tasks.

## Key Features

- **Configuration Parsing**: Utilities (`parser`, `parse_config`) for handling experiment configurations, ensuring flexible and streamlined setup.
- **Fine-Tuning**: Run the main fine-tuning process, delegating tasks to other helper functions.
- Update existing configuration settings on the fly, streamlining adaptation to different tasks.
"""

import argparse
import getpass
import logging
import os
import subprocess
from tkinter.filedialog import askopenfilenames

import appdirs
import numpy as np
import toml
import torch
from PIL import Image

from plantdb.commons import fsdb
from plantdb.commons import io
from romiseg.cli.train_cnn import cnn_train
from romiseg.common.io import model_from_fileset
from romiseg.utils.active_contour import run_refine_romidata

logger = logging.getLogger(__file__)

default_config = "/home/alienor/Documents/scanner-meta-repository/Scan3D/default/segmentation2d_arabidopsis.toml"


def create_folder_if(directory):
    """Creates a folder and an empty file if they do not exist.

    Parameters
    ----------
    directory : str
        The path of the directory to check and create if it does not exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        open('romidb', 'w').close()


def parser():
    """Parses command-line arguments.

    Returns
    -------
    argparse.ArgumentParser
        A configured argument parser with an option for specifying the configuration directory.
    """
    parser = argparse.ArgumentParser(description='Fine-tune a CNN model on a dataset with manually annotated images.')

    parser.add_argument('--config', dest='config', type=str, default=default_config,
                        help='config dir, default: %s' % default_config)

    return parser


def parse_config(config):
    """Parses the configuration file for the fine-tuning and segmentation processes.

    Parameters
    ----------
    config : str
        Path to the configuration TOML file to be parsed.

    Returns
    -------
    str
        Path to the directory containing the weights for fine-tuning.
        If not specified in the configuration, a default cache directory is used.
    str
        Path to the TensorBoard log directory.
        If not specified, a default cache directory is used.
    int
        The batch size for fine-tuning.
    str
        Username for the system.
        If not provided in the configuration, the system's current username is used.
    int
        Number of epochs for the fine-tuning process.
    int
        Size of the input in the X dimension for the model.
    int
        Size of the input in the Y dimension for the model.
    str
        Identifier for the selected model.
    float
        Learning rate for the training process.
    """
    param_pipe = toml.load(str(config))

    direc = param_pipe['Finetune']
    try:
        directory_weights = direc['directory_weights']
    except:
        directory_weights = appdirs.user_cache_dir()
    try:
        tsboard = direc['tsboard'] + '/finetune'
    except:
        tsboard = appdirs.user_cache_dir()
    try:
        user_name = direc['user_name']
    except:
        user_name = getpass.getuser()

    param2 = param_pipe['Segmentation2D']
    # labels = param2['labels']
    Sx = param2['Sx']
    Sy = param2['Sy']
    model_id = param2['model_id']
    finetune = param_pipe['Finetune']
    finetune_epochs = finetune['finetune_epochs']
    batch_size = finetune['batch']
    learning_rate = param2['learning_rate']
    return directory_weights, tsboard, batch_size, user_name, finetune_epochs, Sx, Sy, learning_rate, model_id


def fine_tune_segmentation_model(directory_weights, tsboard, batch_size, user_name, finetune_epochs, Sx, Sy,
                                 learning_rate, model_id):
    """Run the pipeline to fine-tune a segmentation model with manually annotated images.

    This function performs several key steps in a pipeline:
    1. Loads a pre-trained segmentation model and sets it up on the appropriate device (GPU/CPU).
    2. Provides the option to mount a remote dataset from a specified server.
    3. Allows the user to select and annotate images for fine-tuning the model.
    4. Refines the annotated data into a format suitable for training a segmentation model.
    5. Fine-tunes the model on the annotated dataset by freezing backbone layers and training specific layers.
    6. Saves the fine-tuned model for future use.

    Parameters
    ----------
    directory_weights : str
        Path to the directory containing model weights or checkpoint data.
    tsboard : TensorBoard
        TensorBoard object for logging training metrics.
    batch_size : int
        Batch size for fine-tuning the model.
    user_name : str
        Username for accessing the remote dataset.
    finetune_epochs : int
        Number of epochs to fine-tune the model.
    Sx : int
        Spatial dimension size for resizing input images along the x-axis.
    Sy : int
        Spatial dimension size for resizing input images along the y-axis.
    learning_rate : float
        Learning rate for the optimizer during fine-tuning.
    model_id : str
        Unique identifier for the pre-trained model.

    Returns
    -------
    model_name : str
        The name of the fine-tuned model, which includes metadata about its configuration.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model, label_names = save_and_load_model(directory_weights, model_segmentation_name).to(device)

    db_w = fsdb.FSDB(directory_weights)
    db_w.connect()
    s_w = db_w.get_scan('models')
    f_weights = s_w.get_fileset('models')
    model_file = f_weights.get_file(model_id)
    model, label_names = model_from_fileset(model_file)
    model = model.to(device)

    mount_loc = appdirs.user_cache_dir() + '/data_mount/'
    if not os.path.exists(mount_loc):
        os.mkdir(mount_loc)
    txt = subprocess.run(["mountpoint", mount_loc])

    quest = input("Ready to mount romi-project.eu? (y/n) ")
    if quest == 'y':
        subprocess.run(["sshfs", user_name + '@db.romi-project.eu:/data/', mount_loc])

    directory_dataset = mount_loc + '/finetune/'

    files = askopenfilenames(initialdir=os.path.split("/home/")[0],
                             title='Select some pictures to annotate')
    lst = list(files)

    if len(lst) > 0:
        host_scan = files[0].split('/')[-3]
        db = fsdb.FSDB(directory_dataset + '/train/')
        db.connect()

        scan = db.get_scan(host_scan, create=True)
        fileset = scan.get_fileset('images', create=True)
        imgs = np.sort(files)
        label_names = label_names.tolist()
        fileset.set_metadata({'channels': ['rgb'] + label_names})

        for i, path in enumerate(imgs):
            im_name = host_scan + '_' + os.path.split(path)[1][:-4]

            im = np.array(Image.open(path))
            f_im = fileset.create_file(im_name + '_rgb')
            f_im.set_metadata('shot_id', im_name)
            f_im.set_metadata('channel', 'rgb')
            io.write_image(f_im, im, ext='png')

            im_save = fsdb._file_path(f_im)
            subprocess.run(['labelme', im_save, '-O', im_save, '--labels', ','.join(label_names)])

            npz = run_refine_romidata(im_save, 1, 1, 1, 1, 1, class_names=label_names,
                                      plotit=im_save)

            for channel in label_names:
                f_label = fileset.create_file(im_name + '_' + channel)
                f_label.set_metadata('shot_id', im_name)
                f_label.set_metadata('channel', channel)
                io.write_image(f_label, npz[channel], ext='png')
        db.disconnect()

    subprocess.run(["rsync", "-av", directory_dataset, appdirs.user_cache_dir()])
    directory_dataset = appdirs.user_cache_dir()

    # freeze backbone layers
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    model = cnn_train(f_weights, directory_dataset, np.array(label_names), tsboard, batch_size, finetune_epochs,
                      model, Sx, Sy, learning_rate=learning_rate, data_augmentation=False)

    model_name = model_id + os.path.split(directory_dataset)[1] + '_%d_%d_' % (
        Sx, Sy) + 'finetune_epoch%d' % finetune_epochs

    file = f_weights.create_file(model_name)
    io.write_torch(file, model)
    file.set_metadata({'model_id': model_name, 'label_names': label_names.tolist()})

    logger.info("You have fine-tunned the segmentation network with the images you manually annotated.")
    logger.info("The pipeline should work better on your images now, let's launch it again")
    return model_name


def update_toml_config(config, model_name):
    """Updates a TOML configuration file with a new model ID.

    Parameters
    ----------
    config : str
        Path to the TOML configuration file to be updated.
    model_name : str
        The model name to set as the `model_id` in the `Segmentation2D` section of
        the TOML configuration file.

    """
    # Load the existing TOML configuration file into a dictionary
    param_pipe = toml.load(config)
    # Update the 'model_id' in the 'Segmentation2D' section with the new model name
    param_pipe['Segmentation2D']['model_id'] = model_name
    # Convert the updated dictionary back to a TOML formatted string
    text = toml.dumps(param_pipe)
    # Write the updated TOML string back to the configuration file
    with open(config, "w") as f:
        f.write(text)
    return


def main():
    args = parser().parse_args()
    model_name = fine_tune_segmentation_model(*parse_config(args.config))
    update_toml_config(args.config, model_name)


if __name__ == '__main__':
    main()
