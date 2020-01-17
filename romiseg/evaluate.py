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

#import segmentation_models_pytorch as smp
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

from romiseg.utils.train_from_dataset import init_set, Dataset_im_label, train_model, plot_dataset, evaluate, save_and_load_model
from romiseg.utils import segmentation_model


default_config_dir = "/home/alienor/Documents/scanner-meta-repository/Scan3D/config/segmentation2d_arabidopsis.toml"

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
label_names = param2['labels'].split(',')
model_segmentation_name = param2["model_segmentation_name"]



Sx = param2['Sx']
Sy = param2['Sy']

epochs = param2['epochs']
batch_size = param2['batch']

learning_rate = param2['learning_rate']


############################################################################################################################

def cnn_eval(directory_weights, directory_dataset, label_names, tsboard, batch_size, epochs,
                    model, Sx, Sy, load_model = False):
        
    #Training board
    
    #image transformation for training, can be modified for data augmentation
    trans = transforms.Compose([
                                transforms.CenterCrop((Sx, Sy)),
                                transforms.ToTensor(),
                                ])
    
    #Load images and ground truth
    path_test = directory_dataset
    image_test, target_test = init_set('', path_test)

    test_dataset = Dataset_im_label(image_test, target_test, transform = trans)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    with torch.no_grad():
        pred_tot = []
        id_list = []
        count = 0
        print('Image segmentation by the CNN')
    
        for inputs, id_im in test_loader:
            inputs = inputs.to(device) #input image on GPU
            outputs = evaluate(inputs, model)  #output image
            pred_tot.append(outputs)
            id_list.append(id_im)
            count += 1
    

    return pred_tot

    
if __name__ == '__main__':     

    
    '''
    #Load model
    #model = smp.Unet(model_segmentation_name, classes=num_classes, encoder_weights='imagenet').cuda()
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
    
    model = torch.load(directory_weights+'/'+ model_segmentation_name)

    trans = transforms.Compose([
                                transforms.CenterCrop((Sx, Sy)),
                                transforms.ToTensor(),
                                ])
    
    #Load images and ground truth
    path_test = directory_dataset +'/test/'
    image_test, target_test = init_set('', path_test)

    test_dataset = Dataset_im_label(image_test, target_test, transform = trans)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    with torch.no_grad():
        pred_tot = []
        id_list = []
        count = 0
        print('Image segmentation by the CNN')
    
        for inputs, labels in test_loader:
            inputs = inputs.to(device) #input image on GPU
            outputs = evaluate(inputs, model)  #output image
            pred_tot.append(outputs)
            id_list.append(id_im)
            count += 1

    a = pred_tot[0]
    import matplotlib.pyplot as plt
    plt.imshow(a[0,1].cpu().numpy())
    plt.show()