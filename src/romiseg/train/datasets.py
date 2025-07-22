#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset classes for training segmentation models.

This module contains dataset classes used for training segmentation models.
"""

import random
import numpy as np
import torch
from PIL import Image
from plantdb.commons import fsdb
from plantdb.commons import io
from torch.utils.data import Dataset
from torchvision import transforms

from romiseg.common.transforms import ResizeCrop, gaussian, MyRotationTransform


class DatasetImLabel(Dataset):
    """Data handling for Pytorch Dataloader for 2D image segmentation"""

    def __init__(self, shots, channels, size, path, labels=True, data_augmentation=True):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        shots : list
            List of shots to use.
        channels : list
            List of channels to use.
        size : tuple
            Size of the images.
        path : str
            Path to the dataset.
        labels : bool, optional
            Whether to return labels. Default is True.
        data_augmentation : bool, optional
            Whether to use data augmentation. Default is True.
        """
        self.shots = shots
        self.channels = channels
        self.size = size
        self.path = path
        self.get_labels = labels
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        """
        Get an item from the dataset.
        
        Parameters
        ----------
        index : int
            Index of the item to get.
            
        Returns
        -------
        tuple
            Tuple containing the image and labels.
        """
        db = fsdb.FSDB(self.path)
        db.connect()
        db_file_meta = self.shots[index]
        s = db.get_scan(db_file_meta['scan'])
        image_file = s.get_fileset('images').get_files(query={'channel': 'rgb', 'shot_id': db_file_meta['shot_id']})[0]
        # print(db_file_meta['shot_id'])
        image = Image.fromarray(io.read_image(image_file))
        if self.data_augmentation == True:
            angle = random.randint(-90, 90)
            # scale = 1 + np.random.rand()
            padding = image.size
            resize = ResizeCrop(self.size)
            pad = transforms.Pad(padding, padding_mode='reflect')
            crop = transforms.CenterCrop(self.size)
            # scale = transforms.Resize(np.asarray((np.array(self.size) * scale),dtype=int).tolist())
            # id_im = db_file.id
            # rot = MyRotationTransform(angle, fill=0.5)
            # trans = transforms.Compose([resize, scale, pad, rot, crop, transforms.ToTensor()])
            trans1 = transforms.Compose([resize, pad])
            t_image = trans1(image)
            t_image = t_image.rotate(angle, resample=2)
            trans2 = transforms.Compose([crop, transforms.ToTensor()])
            t_image = trans2(t_image)
        else:
            t_image = ResizeCrop(self.size)(image)
            t_image = transforms.ToTensor()(t_image)

        t_image = t_image[0:3, :, :]  # select RGB channels
        t_image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])(t_image)
        dyn = t_image.max()
        if self.data_augmentation == True:
            t_image = gaussian(t_image, is_training=True,
                               mean=0, stddev=np.random.rand() * 1 / 100, dyn=dyn)
        torch_labels = []
        for i, c in enumerate(self.channels):

            labels = s.get_fileset('images').get_files(query={'channel': c, 'shot_id': db_file_meta['shot_id']})[0]
            if c != 'background':
                t_label = Image.fromarray(1.0 * (io.read_image(labels) > 0))
            else:
                t_label = Image.fromarray(1.0 * (io.read_image(labels) == 255))

            num_bands = len(t_label.getbands())
            if self.data_augmentation == True:
                t_label = trans1(t_label)
                t_label = t_label.rotate(angle, resample=2)
                t_label = trans2(t_label)
            else:
                t_label = ResizeCrop(self.size)(t_label)
                t_label = transforms.ToTensor()(t_label)
            torch_labels.append(t_label)
        torch_labels = torch.cat(torch_labels, dim=0)
        db.disconnect()

        return t_image, torch_labels

    def __len__(self):  # return count of sample
        """
        Get the length of the dataset.
        
        Returns
        -------
        int
            Number of items in the dataset.
        """
        return len(self.shots)

    def read_label(self, labels):
        """
        Process labels by adding a background channel.
        
        Parameters
        ----------
        labels : numpy.ndarray
            Labels to process.
            
        Returns
        -------
        numpy.ndarray
            Processed labels.
        """
        somme = labels.sum(axis=0)
        background = somme == 0
        background = background.astype(somme.dtype)
        background = background * 255
        dimx, dimy = background.shape
        background = np.expand_dims(background, axis=0)
        if labels.shape[0] == 5:
            labels = np.concatenate((background * 0, labels), axis=0)

        labels = np.concatenate((background, labels), axis=0)

        return labels


class DatasetImLabel3d(Dataset):
    """Data handling for Pytorch Dataloader for 3D image segmentation"""

    def __init__(self, image_paths, label_paths, voxel_path, transform):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        image_paths : list
            List of paths to images.
        label_paths : list
            List of paths to labels.
        voxel_path : list
            List of paths to voxels.
        transform : callable
            Transform to apply to images and labels.
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.voxel_path = voxel_path
        self.transforms = transform

    def __getitem__(self, index):
        """
        Get an item from the dataset.
        
        Parameters
        ----------
        index : int
            Index of the item to get.
            
        Returns
        -------
        tuple
            Tuple containing the image, labels, and voxel data.
        """
        db_file = self.image_paths[index]
        image = Image.fromarray(io.read_image(db_file))
        # id_im = db_file.id
        t_image = self.transforms(image)  # crop the images
        t_image = t_image[0:3, :, :]  # select RGB channels

        db_file = self.label_paths[index]
        npz = io.read_npz(db_file)
        torch_labels = []

        for i in range(len(npz.files)):
            labels = npz[npz.files[i]]
            # labels = self.read_label(labels)
            t_label = Image.fromarray(np.uint8(labels))
            t_label = self.transforms(t_label)
            torch_labels.append(t_label)
        torch_labels = torch.cat(torch_labels, dim=0)
        somme = torch_labels.sum(dim=0)
        background = somme == 0
        background = background.float()
        background = background
        dimx, dimy = background.shape
        background = background.unsqueeze(0)
        torch_labels = torch.cat((background, torch_labels), dim=0)

        voxel = io.read_torch(self.voxel_path[index])

        return t_image, torch_labels, voxel

    def __len__(self):  # return count of sample
        """
        Get the length of the dataset.
        
        Returns
        -------
        int
            Number of items in the dataset.
        """
        return len(self.image_paths)

    def read_label(self, labels):
        """
        Processes an array of labels by appending a background layer.

        This function computes a background layer where all values are set to 255
        for positions where the sum along the first axis of the input `labels`
        array equals zero. The background layer is then expanded and concatenated
        as the first layer of the input array.

        Parameters
        ----------
        labels : numpy.ndarray
            A multi-dimensional array representing label data. The first
            axis typically represents different label layers.

        Returns
        -------
        numpy.ndarray
            The input labels array with an additional background layer
            concatenated as the first layer.
        """
        somme = labels.sum(axis=0)
        background = somme == 0
        background = background.astype(somme.dtype) * 255
        dimx, dimy = background.shape
        background = np.expand_dims(background, axis=0)
        labels = np.concatenate((background, labels), axis=0)

        return labels