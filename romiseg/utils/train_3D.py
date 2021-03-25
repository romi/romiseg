#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:56:38 2019

@author: alienor
"""


import torch
from torch import autograd
from torch.optim import lr_scheduler
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset

from tqdm import tqdm
from PIL import Image

from romiseg.utils.train_from_dataset import Dataset_im_label, plot_dataset, init_set
import romiseg.utils.alienlab as alien
from romiseg.utils.ply import write_ply

from plantdb import io
from plantdb import fsdb

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from torchvision import transforms

import matplotlib.pyplot as plt
import os
import requests
import copy
import numpy as np


import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##################LOAD PRE-TRAINED WEIGHTS############

def download_file(url, target_dir):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(target_dir + '/' +local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename

def save_and_load_model(weights_folder, model_segmentation_name):


    #if not already saved, download from database 
    if model_segmentation_name not in os.listdir(weights_folder):
        
        url = 'http://db.romi-project.eu/models/' + model_segmentation_name 
        
        download_file(url, weights_folder)
        
    model_segmentation = torch.load(weights_folder + '/' + model_segmentation_name)[0]
    
    try: 
        model_segmentation = model_segmentation.module
    except:
        model_segmentation = model_segmentation
            
    return model_segmentation






def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def my_metric(outputs: torch.Tensor, labels: torch.Tensor):
    inds  = labels != 0
    bools = outputs[inds] == labels[inds]
    
    return torch.mean(bools.float())  # Or thresholded.mean() if you are interested in average across the batch



def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


class classification(torch.nn.Module):
    def __init__(self,D_in,D_out):
        super(classification, self).__init__()
        lin = torch.nn.Linear(D_in,D_in)
        #lin.weight.data = torch.eye(D_in, requires_grad = True).to(device)
        #lin.bias.data.fill_(0)
        self.layer = torch.nn.Sequential(lin)
    def forward(self,x):
        out = self.layer(x)
        return out

def init_set(mode, path):
    db = fsdb.FSDB(path)
    db.connect()
    scans = db.get_scans()
    image_files = []
    gt_files = []
    voxel_files= []
    for s in scans:
        f = s.get_fileset('images')
        list_files = f.files
        shots = [list_files[i].metadata['shot_id'] for i in range(len(list_files))]      
        shots = list(set(shots))
        for shot in shots:
            image_files += f.get_files({'shot_id':shot, 'channel':'rgb'})
            gt_files += f.get_files({'shot_id':shot, 'channel':'segmentation'})
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
        #id_im = db_file.id
        t_image = self.transforms(image) #crop the images
        t_image = t_image[0:3, :, :] #select RGB channels
        
        db_file = self.label_paths[index]
        npz = io.read_npz(db_file)
        torch_labels = []


        for i in range(len(npz.files)):    
            
            labels = npz[npz.files[i]]
            #labels = self.read_label(labels)
            t_label = Image.fromarray(np.uint8(labels))
            t_label = self.transforms(t_label)
            torch_labels.append(t_label)
        torch_labels = torch.cat(torch_labels, dim = 0)
        somme = torch_labels.sum(dim = 0)
        background = somme == 0
        background = background.float()
        background = background
        dimx, dimy = background.shape
        background = background.unsqueeze(0)
        torch_labels = torch.cat((background, torch_labels), dim = 0)
        
        voxel = io.read_torch(self.voxel_path[index])
        
        return t_image, torch_labels, voxel

    def __len__(self):  # return count of sample
        return len(self.image_paths)

    def read_label(self, labels):

        somme = labels.sum(axis = 0)
        background = somme == 0
        background = background.astype(somme.dtype)*255
        dimx, dimy = background.shape
        background = np.expand_dims(background, axis = 0)
        labels = np.concatenate((background, labels), axis = 0)
        
        return labels


def train_model_voxels(train_type, dataloaders, model, optimizer, scheduler, writer, 
                       voxel_loss, torch_voxels, num_epochs=25, viz = False, label_names = []):
    L = []
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    loss_test = []
    n_classes = len(label_names)
    for epoch in range(num_epochs):
        print('Running epoch %d/%d'%(epoch, num_epochs), end="\r")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                #for param_group in optimizer.param_groups:
                #    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
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
                        #print(outputs[1].shape)
                        #pred_class = pred_class[:,:-1]
                        #pred_class = torch.exp(pred_class)

                        loss = calc_loss(outputs[0], labels, metrics) + voxel_loss(pred_class, voxels[0, :, n_classes - 1])
                        #F.cross_entropy(pred_class, voxels[0, :, 3])
                        #print('loss %.15f'%loss)
                    #print(loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        with autograd.detect_anomaly():
                            loss.backward()
                            optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            #print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            L.append(epoch_loss)
            writer.add_scalar('train/crossentropy', epoch_loss, epoch)
        
            if phase == 'val':
                inputs, labels, voxels = next(iter(dataloaders[phase]))
                inputs = inputs.to(device)
                labels = labels.to(device)
                voxels = voxels.to(device)
                lab = torch.argmax(labels, dim = 1)
                # forward
                # track history if only in train
                outputs = model(inputs)

                out = torch.argmax(outputs[0], dim = 1)
                loss_test.append(my_metric(out, lab))
                
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                #print("saving best model")
                best_loss = epoch_loss
                #best_model_wts = copy.deepcopy(model.state_dict())
        
            #plot 4 images to visualize the data
        if viz == True:
 
            plt.ioff()
            fig = plt.figure(figsize = (14, 6))
        
            col = len(label_names)
            for i in range(col):
                plt.subplot(2, col, 2*i + 1)
                plt.axis('off')
                plt.grid(False)
                img = inputs[0]
                img = torchvision.transforms.ToPILImage()(img.detach().cpu())
                plt.imshow(img)
                plt.title('image')
                img = F.sigmoid(outputs[0][0,i,:,:])
                img = torchvision.transforms.ToPILImage()(img.detach().cpu())
                plt.subplot(2, col, 2*i + 2)
                plt.axis('off')
                plt.grid(False)
                plt.imshow(img)
                plt.title(label_names[i])
            
            
            
            writer.add_figure('Segmented images', fig, epoch)
            
            
            colors = ['r.', 'k.', 'g.', 'b.', 'o.', 'l.']

            fig = plt.figure(figsize = (10,10))
            ax = plt.axes(projection='3d')
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.set_zlim(-60, 60)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            #ax.set_aspect('equal')
            ax.set_title('Ground truth predictions')
            pred_class = outputs[1]#[:,:-1]
            #pred_class = torch.exp(pred_class)
            #preds_max = torch.max(pred_class, dim = -1).values
            pred_class = torch.argmax(pred_class, dim = -1)
            for i, label in enumerate(label_names):
                if i != 0:                    
                                        
                    inds = voxels[0, :, 3] == i
                    inds = inds.cpu()
                    print('ground truth ', label, np.count_nonzero(inds))
                    pred_label_gt = torch_voxels[inds].detach().cpu()
                    ax.scatter3D(pred_label_gt[:,0], pred_label_gt[:,1], pred_label_gt[:,2], colors[2], s=10)

                    inds = (pred_class == i)
                    inds = inds.cpu()
                    print('prediction ', label, np.count_nonzero(inds))
                    pred_label = torch_voxels[inds].detach().cpu()
                    ax.scatter3D(pred_label[:,0], pred_label[:,1], pred_label[:,2], colors[1], s=10)


            #print(model.class_layer[0].weight.data, model.class_layer[0].bias.data)
            torch_voxels[:,3] = 0
            torch_voxels[:,3] = pred_class
            writer.add_figure('Segmented point cloud', fig, epoch)
            voxels_class = torch_voxels[(torch_voxels[:,3] != 0)*(torch_voxels[:,3] != len(label_names))]
            write_ply('/home/alienor/Documents/training2D/volume/training_epoch_%d'%epoch, voxels_class.detach().cpu().numpy(), ['x', 'y', 'z', 'labels'])    
        
        #time_elapsed = time.time() - since
        #print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    #print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
        #model.load_state_dict(best_model_wts)
        
    torch_voxels[:,3] = 0
    torch_voxels[:,3] = voxels[0, :,3]
    writer.add_figure('Segmented point cloud', fig, epoch)
    voxels_class = torch_voxels[(torch_voxels[:,3] != 0)*(torch_voxels[:,3] != len(label_names))]
    write_ply('/home/alienor/Documents/training2D/volume/ground_truth', voxels_class.detach().cpu().numpy(), ['x', 'y', 'z', 'labels'])    
    
    return model, L, loss_test






def fine_tune_train(path_train, path_val, weights_folder, label_names, tsboard_name,
                    model_segmentation_name, Sx, Sy, num_epochs, scan):
    num_classes = len(label_names)
    
    trans = transforms.Compose([
    transforms.CenterCrop((Sx, Sy)),
    transforms.ToTensor(),
])
    
    image_train, target_train = init_set('', path_train, 'jpg')
    image_val, target_val = init_set('', path_val, 'jpg')

    train_dataset = Dataset_im_label(image_train, target_train, transform = trans)
    val_dataset = Dataset_im_label(image_val, target_val, transform = trans) 
    
        
    batch_size = min(num_classes, len(image_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    
    plot_dataset(train_loader, label_names, batch_size) #display training set
    
    print('Now the network will train on the data you annotated')

    
    batch_size = 2
       
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        }
    
    
    model = save_and_load_model(weights_folder, model_segmentation_name)

    
    writer = SummaryWriter('test')#tsboard_name)


    a = list(model.children())
    for child in  a[0].children():
        for param in child.parameters():
            param.requires_grad = False
    
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    
    
    model = train_model(dataloaders, model, optimizer_ft, exp_lr_scheduler, writer,  num_epochs = num_epochs)
    ext_name = '_finetune_' + scan + '_epoch%d.pt'%num_epochs
    new_model_name = model_segmentation_name[:-3] + ext_name

    torch.save(model, weights_folder + '/' + new_model_name)
    
    
    return model, new_model_name




    
# Prediction
def evaluate(inputs, model):

    with torch.no_grad():
        inputs.requires_grad = False
        # Get the first batch
        inputs = inputs.to(device)

        pred = model(inputs)
        # The loss functions include the sigmoid function.
        pred = F.sigmoid(pred)
        
    return pred
    
