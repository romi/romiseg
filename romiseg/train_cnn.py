#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:18:24 2019

@author: alienor
"""


import os
import argparse
import toml
from PIL import Image
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import torch.optim as optim
#from torchvision import models

from romidata import io
from romidata import fsdb

from romiseg.utils.train_from_dataset import init_set, Dataset_im_label, train_model, plot_dataset, save_and_load_model
from romiseg.utils import segmentation_model


default_config_dir = "/home/alienor/Documents/scanner-meta-repository/Scan3D/default/segmentation2d_arabidopsis.toml"

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--config', dest='config', default=default_config_dir,
                    help='config dir, default: %s'%default_config_dir)


args = parser.parse_args()


param_pipe = toml.load(args.config)

direc = param_pipe['TrainingDirectory']

path = direc['path']
directory_weights = path + direc['directory_weights']
tsboard = path +  direc['tsboard'] + '/2D_segmentation'
directory_dataset = path + direc['directory_dataset']


param2 = param_pipe['Segmentation2D']
model_name = param2["model_name"]
model_segmentation_name = param2["model_segmentation_name"]

#label_names = param2['labels'].split(',')


Sx = param2['Sx']
Sy = param2['Sy']

epochs = param2['epochs']
batch_size = param2['batch']

learning_rate = param2['learning_rate']


############################################################################################################################


def cnn_train(directory_weights, directory_dataset, label_names, tsboard, batch_size, epochs,
                    model, Sx, Sy, load_model = False, showit = False):
        
    #Training board
    writer = SummaryWriter(tsboard)
    
    #image transformation for training, can be modified for data augmentation
    trans = transforms.Compose([
                                transforms.CenterCrop((Sx, Sy)),
                                transforms.ToTensor(),
                                ])
    
    #Load images and ground truth
    path_val = directory_dataset + '/val/'
    path_train = directory_dataset + '/train/'
    path_test = directory_dataset + '/test/'
    
    image_train, channels = init_set('', path_train)
    image_val, channels = init_set('', path_val)
    image_test, channels = init_set('', path_test)
    
    train_dataset = Dataset_im_label(image_train, channels, transform = trans, path = path_train)
    val_dataset = Dataset_im_label(image_val, channels, transform = trans, path = path_train) 
    test_dataset = Dataset_im_label(image_test, channels, transform = trans, path = path_train)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    #Show input images 
    fig = plot_dataset(train_loader, label_names, batch_size, showit = True) #display training set
    writer.add_figure('Dataset images', fig, 0)
    
       
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
       }


    
    #Choice of optimizer, can be changed
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    #make learning rate evolve
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    
    #Run training
    model = train_model(dataloaders, model, optimizer_ft, exp_lr_scheduler, writer, 
                        num_epochs = epochs, viz = True, label_names = label_names)
    #save model

    return model

#######
    
if __name__ == '__main__':     

    
    '''
<<<<<<< Updated upstream
    #Load model

=======
    #Load model   
>>>>>>> Stashed changes
    #model = models.segmentation.fcn_resnet101(pretrained=True)
    #model = torch.nn.Sequential(model, torch.nn.Linear(21, num_classes)).cuda()
    
      
    #Freeze encoder
    a = list(model.children())
    for child in  a[0].children():
        for param in child.parameters():
            param.requires_grad = False
    '''
       
    num_classes = len(label_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = segmentation_model.ResNetUNet(num_classes).to(device)
   # model = save_and_load_model(directory_weights, model_segmentation_name)
                                
    # freeze backbone layers
    for l in model.base_layers:
        for param in l.parameters():
            param.requires_grad = False
   

    model = cnn_train(directory_weights, directory_dataset, label_names, tsboard, batch_size, epochs,
                     model, Sx, Sy)

    model_name =  model_name + os.path.split(directory_dataset)[1] +'_%d_%d'%(Sx,Sy)+ '_epoch%d.pt'%epochs
    
    torch.save(model, directory_weights + '/' + model_name)
    
    
    
