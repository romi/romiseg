#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Training Functions.

This module contains functions for training segmentation models.
"""

import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from romiseg.common.io import save_and_load_model
from romiseg.common.dataset import init_set
from romiseg.common.visualization import plot_dataset
from romiseg.train.losses import calc_loss
from romiseg.train.metrics import print_metrics, my_metric
from romiseg.train.datasets import DatasetImLabel

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(f_weights, dataloaders, model, optimizer, scheduler, writer, num_epochs=25, viz=False, label_names=[]):
    """
    Train a segmentation model.
    
    Parameters
    ----------
    f_weights : plantdb.FSDB.Fileset
        Fileset to save model weights to.
    dataloaders : dict
        Dictionary containing DataLoader objects for 'train' and 'val' phases.
    model : torch.nn.Module
        Model to train.
    optimizer : torch.optim.Optimizer
        Optimizer to use.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    writer : torch.utils.tensorboard.SummaryWriter
        TensorBoard writer.
    num_epochs : int, optional
        Number of epochs to train for. Default is 25.
    viz : bool, optional
        Whether to visualize training progress. Default is False.
    label_names : list, optional
        List of label names. Default is an empty list.
        
    Returns
    -------
    tuple
        Tuple containing the trained model and a dictionary of training metrics.
    """
    L = {'bce': [], 'dice': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    loss_test = []
    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        print('Running epoch %d/%d' % (epoch, num_epochs), end="\r")

        # since = time.time()

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

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    # print(loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # print(model.conv_last.weight.grad, model.conv_last.bias.grad)
                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            L['bce'].append(metrics['bce'] / epoch_samples)
            L['dice'].append(metrics['dice'] / epoch_samples)
            writer.add_scalar('train/crossentropy', epoch_loss, epoch)

            if phase == 'val':
                inputs, labels = next(iter(dataloaders[phase]))
                inputs = inputs.to(device)
                labels = labels.to(device)
                lab = torch.argmax(labels, dim=1)
                # forward
                # track history if only in train
                outputs = model(inputs)
                out = torch.argmax(outputs, dim=1)
                # loss_test.append(my_metric(out, lab))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                # print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

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
                img = F.sigmoid(outputs[0, i, :, :])
                img = torchvision.transforms.ToPILImage()(img.detach().cpu())
                plt.subplot(2, col, 2 * i + 2)
                plt.axis('off')
                plt.grid(False)
                plt.imshow(img)
                plt.title(label_names[i])

            writer.add_figure('Segmented images', fig, epoch)
        if epoch % 10 == 0:
            model_name = 'tmp_epoch%d' % epoch

            file = f_weights.create_file(model_name)
            from plantdb.commons import io
            io.write_torch(file, model)
            file.set_metadata({'model_id': model_name, 'label_names': label_names.tolist()})

        # time_elapsed = time.time() - since
        # print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights 
    model.load_state_dict(best_model_wts)
    return model, L  # , loss_test


def fine_tune_train(path_train, path_val, weights_folder, label_names, tsboard_name,
                    model_segmentation_name, Sx, Sy, num_epochs, scan):
    """
    Fine-tune a pre-trained segmentation model.
    
    Parameters
    ----------
    path_train : str
        Path to the training dataset.
    path_val : str
        Path to the validation dataset.
    weights_folder : str
        Path to the folder containing model weights.
    label_names : list
        List of label names.
    tsboard_name : str
        Name for TensorBoard logs.
    model_segmentation_name : str
        Name of the pre-trained model file.
    Sx : int
        Width of the input images.
    Sy : int
        Height of the input images.
    num_epochs : int
        Number of epochs to train for.
    scan : str
        Scan identifier.
        
    Returns
    -------
    tuple
        Tuple containing the trained model and the name of the saved model file.
    """
    num_classes = len(label_names)

    image_train, target_train = init_set('', path_train)
    image_val, target_val = init_set('', path_val)

    train_dataset = DatasetImLabel(image_train, target_train, (Sx, Sy), path=path_train)
    val_dataset = DatasetImLabel(image_val, target_val, (Sx, Sy), path=path_val)

    batch_size = min(num_classes, len(image_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    fig = plot_dataset(train_loader, label_names, batch_size)  # display training set
    plt.show(block=True)
    print('Now the network will train on the data you annotated')

    batch_size = 2

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    model = save_and_load_model(weights_folder, model_segmentation_name, label_names)

    writer = SummaryWriter('test')  # tsboard_name)

    a = list(model.children())
    for child in a[0].children():
        for param in child.parameters():
            param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model, _ = train_model(None, dataloaders, model, optimizer_ft, exp_lr_scheduler, writer, num_epochs=num_epochs)
    ext_name = '_finetune_' + scan + '_epoch%d.pt' % num_epochs
    new_model_name = model_segmentation_name[:-3] + ext_name

    torch.save(model, weights_folder + '/' + new_model_name)

    return model, new_model_name