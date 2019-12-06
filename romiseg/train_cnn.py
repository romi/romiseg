#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:18:24 2019

@author: alienor
"""
import open3d

import os
import argparse
import toml
from PIL import Image
import numpy as np

import segmentation_models_pytorch as smp
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import torch.optim as optim
#from torchvision import models

from romidata import io
from romidata import fsdb

from romiseg.utils.train_from_dataset import train_model
from romiseg.utils.dataloader_finetune import plot_dataset
from romiseg.utils import segmentation_model

default_config_dir = "romiseg/parameters_train.toml"

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--config', dest='config', default=default_config_dir,
                    help='config dir, default: %s'%default_config_dir)


args = parser.parse_args()



param_pipe = toml.load(args.config)

param = param_pipe['Segmentation2D']
path = param['path']
directory_weights = path + param['directory_weights']
model_segmentation_name = param['model_segmentation_name']
tsboard = path +  param['tsboard']

label_names = param['labels'].split(',')

directory_dataset = path + param['directory_dataset']


Sx = param['Sx']
Sy = param['Sy']

epochs = param['epochs']
batch_size = param['batch']

learning_rate = param['learning_rate']



############################################################################################################################

def init_set(mode, path):
    db = fsdb.FSDB(path)
    db.connect()
    scans = db.get_scans()
    image_files = []
    gt_files = []
    for s in scans:
        f = s.get_fileset('images')
        list_files = f.files
        shots = [list_files[i].metadata['shot_id'] for i in range(len(list_files))]      
        shots = list(set(shots))
        for shot in shots:
            image_files += f.get_files({'shot_id':shot, 'channel':'rgb'})
            gt_files += f.get_files({'shot_id':shot, 'channel':'segmentation'})

    db.disconnect()
    return image_files, gt_files



class Dataset_im_label(Dataset): 
    """Data handling for Pytorch Dataloader"""

    def __init__(self, image_paths, label_paths, transform):  

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transform

    def __getitem__(self, index):

        db_file = self.image_paths[index]
        image = Image.fromarray(io.read_image(db_file))
        #id_im = db_file.id
        t_image = self.transforms(image) #crop the images
        t_image = t_image[0:3, :, :] #select RGB channels
        
        db_file = self.label_paths[index]
        npz = io.read_npz(db_file)
        labels = npz[npz.files[0]]
        labels = self.read_label(labels)
        torch_labels = []
        for i in range(labels.shape[0]):
            t_label = Image.fromarray(np.uint8(labels[i]))
            t_label = self.transforms(t_label)
            torch_labels.append(t_label)
        torch_labels = torch.cat(torch_labels, dim = 0)
        return t_image, torch_labels

    def __len__(self):  # return count of sample
        return len(self.image_paths)

    def read_label(self, labels):

        somme = labels.sum(axis = 0)
        background = somme == 0
        background = background.astype(somme.dtype)
        dimx, dimy = background.shape
        background = np.expand_dims(background, axis = 0)
        labels = np.concatenate((background, labels), axis = 0)
        
        return labels


#def cnn_train(directory_weights, directory_dataset, label_names, tsboard, batch_size, epochs,
#                    model_segmentation_name, Sx, Sy):
    
#Training board
writer = SummaryWriter(tsboard)
num_classes = len(label_names)

#image transformation for training, can be modified for data augmentation
trans = transforms.Compose([
                            transforms.CenterCrop((Sx, Sy)),
                            transforms.ToTensor(),
                            ])

#Load images and ground truth
path_val = directory_dataset# + '/val/'
path_train = directory_dataset# + '/train/'

image_train, target_train = init_set('', path_train)
image_val, target_val = init_set('', path_val)
    
train_dataset = Dataset_im_label(image_train, target_train, transform = trans)
val_dataset = Dataset_im_label(image_val, target_val, transform = trans) 
        
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

#Show input images 
fig = plot_dataset(train_loader, label_names, batch_size, showit = False) #display training set
writer.add_figure('Dataset images', fig, 0)

   
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    }
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Load model
model = smp.Unet(model_segmentation_name, classes=num_classes, encoder_weights='imagenet').cuda()
#model = models.segmentation.fcn_resnet101(pretrained=True)
#model = torch.nn.Sequential(model, torch.nn.Linear(21, num_classes)).cuda()

  
#Freeze encoder
a = list(model.children())
for child in  a[0].children():
    for param in child.parameters():
        param.requires_grad = False
'''
   

      
model = segmentation_model.ResNetUNet(num_classes).to(device)

# freeze backbone layers
for l in model.base_layers:
    for param in l.parameters():
        param.requires_grad = False
   
'''
#Choice of optimizer, can be changed
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
#make learning rate evolve
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

#Run training
model = train_model(dataloaders, model, optimizer_ft, exp_lr_scheduler, writer, 
                    num_epochs = epochs, viz = True, label_names = label_names)
#save model
model_name =  model_segmentation_name + os.path.split(directory_dataset)[1] + '_epoch%d.pt'%epochs
torch.save(model, directory_weights + '/' + model_name)

'''
    return model, model_name

#######
cnn_train(directory_weights, directory_dataset, label_names, tsboard, batch_size, epochs,
                    model_segmentation_name, Sx, Sy)
'''