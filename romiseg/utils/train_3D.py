#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:56:38 2019

@author: alienor
"""

import os
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torch import autograd
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from plantdb import fsdb
from plantdb import io
from romiseg.utils.ply import write_ply
from romiseg.utils.train_from_dataset import Dataset_im_label
from romiseg.utils.train_from_dataset import init_set
from romiseg.utils.train_from_dataset import plot_dataset

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


##################LOAD PRE-TRAINED WEIGHTS############

def download_file(url, target_dir):
    """Downloads a file from the given URL and saves it to the specified target directory.

    The function reads the file in chunks to handle large files efficiently and saves it with the same name as in
    the URL. The downloaded file is streamed to avoid loading the whole file into memory.

    Parameters
    ----------
    url : str
        The URL of the file to be downloaded.
    target_dir : str
        The directory where the downloaded file will be saved. The file will retain its name from
        the URL.

    Returns
    -------
    str
        The name of the downloaded file.
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
    """Downloads and loads a neural network model for segmentation.

    Parameters
    ----------
    weights_folder : str
        The directory path where model weights are stored or will be downloaded to.
    model_segmentation_name : str
        The name of the segmentation model file (including its extension).

    Returns
    -------
    torch.nn.Module
        The loaded segmentation model.
    """
    # if not already saved, download from database
    if model_segmentation_name not in os.listdir(weights_folder):
        url = 'http://db.romi-project.eu/models/' + model_segmentation_name

        download_file(url, weights_folder)

    model_segmentation = torch.load(weights_folder + '/' + model_segmentation_name)[0]

    try:
        model_segmentation = model_segmentation.module
    except:
        model_segmentation = model_segmentation

    return model_segmentation


def dice_loss(pred, target, smooth=1.):
    """
    Calculates the Dice loss, a common loss function used for image segmentation tasks, which maximizes the overlap
    between predicted and target binary masks. The Dice loss can handle imbalanced datasets by focusing on the
    overlap rather than absolute pixel accuracy.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted binary mask, typically the model's output, expected to have the shape
        (batch_size, channels, height, width).

    target : torch.Tensor
        Ground truth binary mask, expected to have the same shape as `pred`.

    smooth : float, optional
        A smoothing factor to avoid division by zero or undefined values, default is 1.

    Returns
    -------
    torch.Tensor
        The scalar value representing the mean Dice loss for all the batches in the input.
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def my_metric(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Computes the average correspondence between predicted and target outputs, excluding
    entries where the label is zero. This function effectively calculates a filtered
    accuracy metric, considering only valid labels and comparing them against the
    corresponding outputs.

    Parameters
    ----------
    outputs : torch.Tensor
        The predicted outputs, typically the result of a model's forward pass.
    labels : torch.Tensor
        The ground truth labels against which predictions are compared, with `0`
        indicating labels to exclude from evaluation.

    Returns
    -------
    torch.Tensor
        A single-element tensor containing the computed average correspondence (mean
        accuracy) for non-zero labels. The value lies in the range [0, 1], where 1
        indicates perfect correspondence.
    """
    inds = labels != 0
    bools = outputs[inds] == labels[inds]

    return torch.mean(bools.float())  # Or thresholded.mean() if you are interested in average across the batch


def calc_loss(pred, target, metrics, bce_weight=0.5):
    """
    Computes a weighted loss combining Binary Cross Entropy (BCE) and Dice loss.

    This function calculates the BCE loss and the Dice loss, incorporates a
    weighting factor between the two, and tracks the metrics for each category.
    The loss is computed per batch, and the metrics dictionary is updated with
    the cumulative loss values.

    Parameters
    ----------
    pred : torch.Tensor
        The predicted logits from the model. Should have the same shape as
        `target`.
    target : torch.Tensor
        The ground truth binary labels. Should have the same shape as `pred`.
    metrics : dict
        A dictionary to store cumulative loss values for BCE, Dice, and overall
        loss. This dictionary is updated in-place.
    bce_weight : float, optional
        The weight to balance BCE loss and Dice loss. Defaults to 0.5, with equal
        emphasis on both losses.

    Returns
    -------
    torch.Tensor
        The weighted loss value for the batch, combining BCE and Dice losses.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    """
    Logs and prints the evaluation metrics for a specific phase of training.

    The function computes the average value of each metric over the total number
    of samples for the given phase (e.g., training or validation) and formats the
    outputs for display.

    Parameters
    ----------
    metrics : dict
        A dictionary containing metric names as keys and their corresponding
        cumulative values as values. These represent metrics calculated during
        the current epoch.

    epoch_samples : int
        Total number of samples for which the metrics in `metrics` were
        accumulated during the current phase of the epoch.

    phase : str
        Specifies the current phase of training (e.g., 'train', 'val'). Used
        to format the printed output.

    """
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


class classification(torch.nn.Module):
    """Neural network module for classification tasks.

    It initializes a sequence of layers, consisting of a single linear layer, and
    defines a forward pass through the network. The input dimensionality must
    match the dimension provided during initialization.

    Attributes
    ----------
    layer : torch.nn.Sequential
        The sequential container holding the layers of the neural network.
    """
    def __init__(self, D_in, D_out):
        super(classification, self).__init__()
        lin = torch.nn.Linear(D_in, D_in)
        # lin.weight.data = torch.eye(D_in, requires_grad = True).to(device)
        # lin.bias.data.fill_(0)
        self.layer = torch.nn.Sequential(lin)

    def forward(self, x):
        """Implements the forward propagation for a neural network layer.

        This method applies the computations of a neural network layer
        using the input tensor `x` and returns the output tensor `out`.
        The layer computation logic should be defined in `self.layer`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the layer. The specific shape and dtype of
            the tensor depend on the particular application and model architecture.

        Returns
        -------
        out : torch.Tensor
            Output tensor after applying the layer's computation logic
            on the input tensor `x`. The shape and dtype of the output tensor
            depend on the specifics of `self.layer` processing the input.
        """
        out = self.layer(x)
        return out


def init_set(mode, path):
    """
    Initializes and retrieves datasets for a specific mode and file path. It connects to the file system database,
    fetches the relevant scans, and organizes the required files (images, ground truth files, and voxel files)
    into distinct lists. The result includes files for specific criteria such as shot ID and channel.

    Parameters
    ----------
    mode : str
        The mode that determines how datasets are initialized.
    path : str
        The path to the file system database where the scans are stored.

    Returns
    -------
    tuple of lists
        A tuple containing three lists:
        - List of image files with channel `rgb` for each unique shot ID.
        - List of ground truth files with channel `segmentation` for each unique shot ID.
        - List of voxel files from the `ground_truth_3D` fileset.
    """
    db = fsdb.FSDB(path)
    db.connect()
    scans = db.get_scans()
    image_files = []
    gt_files = []
    voxel_files = []
    for s in scans:
        f = s.get_fileset('images')
        list_files = f.files
        shots = [list_files[i].metadata['shot_id'] for i in range(len(list_files))]
        shots = list(set(shots))
        for shot in shots:
            image_files += f.get_files({'shot_id': shot, 'channel': 'rgb'})
            gt_files += f.get_files({'shot_id': shot, 'channel': 'segmentation'})
            v = s.get_fileset('ground_truth_3D')
            voxel_files += v.get_files()
    db.disconnect()
    return image_files, gt_files, voxel_files


class Dataset_im_label_3D(Dataset):
    """Data handling for Pytorch Dataloader"""

    def __init__(self, image_paths, label_paths, voxel_path, transform):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.voxel_path = voxel_path
        self.transforms = transform

    def __getitem__(self, index):
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
        return len(self.image_paths)

    def read_label(self, labels):
        """Processes an array of labels by appending a background layer.

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


def fine_tune_train(path_train, path_val, weights_folder, label_names, tsboard_name,
                    model_segmentation_name, Sx, Sy, num_epochs, scan):
    """
    Fine-tunes and trains a segmentation model with a given dataset.

    This function receives training and validation dataset paths, applies transformations,
    creates data loaders, and fine-tunes a predefined segmentation model. The segmentation model
    is trained for a specified number of epochs. The model is saved after training for further use.

    Parameters
    ----------
    path_train : str
        The path to the directory containing the training images and labels.
    path_val : str
        The path to the directory containing the validation images and labels.
    weights_folder : str
        The directory where the fine-tuned model weights will be saved.
    label_names : list of str
        The list of class label names corresponding to the dataset.
    tsboard_name : str
        The name for the TensorBoard summary directory or output specification.
    model_segmentation_name : str
        The file name of the pre-trained segmentation model weights to be loaded.
    Sx : int
        The width (in pixels) to which all input images will be cropped for training.
    Sy : int
        The height (in pixels) to which all input images will be cropped for training.
    num_epochs : int
        The number of epochs for which the model will be trained.
    scan : str
        A custom identifier for naming the fine-tuned model during saving.

    Returns
    -------
    model : torch.nn.Module
        The fine-tuned segmentation model trained on the given dataset.
    new_model_name : str
        The file name of the fine-tuned model saved in the specified weights folder.
    """
    num_classes = len(label_names)

    trans = transforms.Compose([
        transforms.CenterCrop((Sx, Sy)),
        transforms.ToTensor(),
    ])

    image_train, target_train = init_set('', path_train, 'jpg')
    image_val, target_val = init_set('', path_val, 'jpg')

    train_dataset = Dataset_im_label(image_train, target_train, transform=trans)
    val_dataset = Dataset_im_label(image_val, target_val, transform=trans)

    batch_size = min(num_classes, len(image_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    plot_dataset(train_loader, label_names, batch_size)  # display training set

    print('Now the network will train on the data you annotated')

    batch_size = 2

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    model = save_and_load_model(weights_folder, model_segmentation_name)

    writer = SummaryWriter('test')  # tsboard_name)

    a = list(model.children())
    for child in a[0].children():
        for param in child.parameters():
            param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(dataloaders, model, optimizer_ft, exp_lr_scheduler, writer, num_epochs=num_epochs)
    ext_name = '_finetune_' + scan + '_epoch%d.pt' % num_epochs
    new_model_name = model_segmentation_name[:-3] + ext_name

    torch.save(model, weights_folder + '/' + new_model_name)

    return model, new_model_name


# Prediction
def evaluate(inputs, model):
    """
    Evaluates a model's predictions for the given input data.

    This function disables gradient computation to improve efficiency during
    evaluation. It processes the input data by transferring it to the required
    device, performs a forward pass through the model, and applies a sigmoid
    function to the predictions. The sigmoid function maps the prediction values
    to probabilities in the range of 0 to 1.

    Parameters
    ----------
    inputs : torch.Tensor
        The input tensor to be evaluated by the model. This tensor should already
        be preprocessed and ready to be passed into the model.
    model : torch.nn.Module
        The PyTorch model to be used for making predictions. The model should
        already be loaded onto the required device.

    Returns
    -------
    torch.Tensor
        The predictions for the input data after applying the sigmoid function.
    """
    with torch.no_grad():
        inputs.requires_grad = False
        # Get the first batch
        inputs = inputs.to(device)

        pred = model(inputs)
        # The loss functions include the sigmoid function.
        pred = F.sigmoid(pred)

    return pred
