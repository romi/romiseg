#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D-specific training functionality for segmentation models.

This module contains functions for training 3D segmentation models.
"""

import copy
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torch import autograd
from tqdm import tqdm

from romiseg.train.losses import calc_loss
from romiseg.train.metrics import print_metrics, my_metric
from romiseg.utils.ply import write_ply

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model_voxels(train_type, dataloaders, model, optimizer, scheduler, writer,
                       voxel_loss, torch_voxels, num_epochs=25, viz=False, label_names=[]):
    """Trains a voxel-based deep learning model using the provided training and validation data.

    The function supports training both a Segmentation model and a Full-pipeline model.
    During the training process, metrics are calculated for evaluation, datasets are iteratively
    processed, loss values are optimized, and visualizations of results can be logged. The best
    model weights can be captured based on validation loss during the process.

    Parameters
    ----------
    train_type : str
        Type of training to perform. Options are 'Segmentation' or 'Fullpipe'.
    dataloaders : dict
        Dictionary containing DataLoader objects for 'train' and 'val' phases.
    model : torch.nn.Module
        The deep learning model to train.
    optimizer : torch.optim.Optimizer
        Optimizer instance to optimize the model parameters.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler to adjust the learning rate during training.
    writer : torch.utils.tensorboard.SummaryWriter
        TensorBoard summary writer for logging metrics and visualizations.
    voxel_loss : Callable
        Loss function used for voxel predictions during training.
    torch_voxels : torch.Tensor
        Input tensor containing voxel data, including ground truth class labels.
    num_epochs : int, optional
        Number of epochs to train the model. Default is 25.
    viz : bool, optional
        Flag to enable or disable result visualization during training. Default is False.
    label_names : list of str, optional
        List of label names corresponding to classes in the dataset. Default is an empty list.

    Returns
    -------
    tuple
        A tuple containing the following:
        - model (torch.nn.Module): Trained model.
        - L (list): List of loss values logged at each epoch.
        - loss_test (list): List of metrics calculated during validation phases.
    """
    L = []
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    loss_test = []
    n_classes = len(label_names)
    for epoch in range(num_epochs):
        print('Running epoch %d/%d' % (epoch, num_epochs), end="\r")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                # for param_group in optimizer.param_groups:
                #    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels, voxels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                voxels = voxels.long().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    pred_class = outputs[1]
                    if train_type == 'Segmentation':
                        loss = calc_loss(outputs[0], labels, metrics)
                    if train_type == 'Fullpipe':
                        # print(outputs[1].shape)
                        # pred_class = pred_class[:,:-1]
                        # pred_class = torch.exp(pred_class)

                        loss = calc_loss(outputs[0], labels, metrics) + voxel_loss(pred_class,
                                                                                   voxels[0, :, n_classes - 1])
                        # F.cross_entropy(pred_class, voxels[0, :, 3])
                        # print('loss %.15f'%loss)
                    # print(loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        with autograd.detect_anomaly():
                            loss.backward()
                            optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            # print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            L.append(epoch_loss)
            writer.add_scalar('train/crossentropy', epoch_loss, epoch)

            if phase == 'val':
                inputs, labels, voxels = next(iter(dataloaders[phase]))
                inputs = inputs.to(device)
                labels = labels.to(device)
                voxels = voxels.to(device)
                lab = torch.argmax(labels, dim=1)
                # forward
                # track history if only in train
                outputs = model(inputs)
                out = torch.argmax(outputs[0], dim=1)
                loss_test.append(my_metric(out, lab))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                # print("saving best model")
                best_loss = epoch_loss
                # best_model_wts = copy.deepcopy(model.state_dict())

            # plot 4 images to visualize the data
        if viz == True:

            plt.ioff()
            fig = plt.figure(figsize=(14, 6))

            col = len(label_names)
            for i in range(col):
                plt.subplot(2, col, 2 * i + 1)
                plt.axis('off')
                plt.grid(False)
                img = inputs[0]
                img = torchvision.transforms.ToPILImage()(img.detach().cpu())
                plt.imshow(img)
                plt.title('image')
                img = F.sigmoid(outputs[0][0, i, :, :])
                img = torchvision.transforms.ToPILImage()(img.detach().cpu())
                plt.subplot(2, col, 2 * i + 2)
                plt.axis('off')
                plt.grid(False)
                plt.imshow(img)
                plt.title(label_names[i])

            writer.add_figure('Segmented images', fig, epoch)

            colors = ['r.', 'k.', 'g.', 'b.', 'o.', 'l.']

            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.set_zlim(-60, 60)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            # ax.set_aspect('equal')
            ax.set_title('Ground truth predictions')
            pred_class = outputs[1]  # [:,:-1]
            # pred_class = torch.exp(pred_class)
            # preds_max = torch.max(pred_class, dim = -1).values
            pred_class = torch.argmax(pred_class, dim=-1)
            for i, label in enumerate(label_names):
                if i != 0:
                    inds = voxels[0, :, 3] == i
                    inds = inds.cpu()
                    print('ground truth ', label, np.count_nonzero(inds))
                    pred_label_gt = torch_voxels[inds].detach().cpu()
                    ax.scatter3D(pred_label_gt[:, 0], pred_label_gt[:, 1], pred_label_gt[:, 2], colors[2], s=10)

                    inds = (pred_class == i)
                    inds = inds.cpu()
                    print('prediction ', label, np.count_nonzero(inds))
                    pred_label = torch_voxels[inds].detach().cpu()
                    ax.scatter3D(pred_label[:, 0], pred_label[:, 1], pred_label[:, 2], colors[1], s=10)

            # print(model.class_layer[0].weight.data, model.class_layer[0].bias.data)
            torch_voxels[:, 3] = 0
            torch_voxels[:, 3] = pred_class
            writer.add_figure('Segmented point cloud', fig, epoch)
            voxels_class = torch_voxels[(torch_voxels[:, 3] != 0) * (torch_voxels[:, 3] != len(label_names))]
            write_ply('/home/alienor/Documents/training2D/volume/training_epoch_%d' % epoch,
                      voxels_class.detach().cpu().numpy(), ['x', 'y', 'z', 'labels'])

            # time_elapsed = time.time() - since
        # print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    # model.load_state_dict(best_model_wts)

    torch_voxels[:, 3] = 0
    torch_voxels[:, 3] = voxels[0, :, 3]
    writer.add_figure('Segmented point cloud', fig, epoch)
    voxels_class = torch_voxels[(torch_voxels[:, 3] != 0) * (torch_voxels[:, 3] != len(label_names))]
    write_ply('/home/alienor/Documents/training2D/volume/ground_truth', voxels_class.detach().cpu().numpy(),
              ['x', 'y', 'z', 'labels'])

    return model, L, loss_test