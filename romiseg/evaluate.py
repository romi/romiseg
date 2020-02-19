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
import matplotlib.pyplot as plt


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

from numpy import random
from romidata import io
from romidata import fsdb

from tqdm import tqdm
from romiseg.utils.train_from_dataset import init_set, Dataset_im_label, train_model, plot_dataset, test, save_and_load_model
from romiseg.utils import segmentation_model
import romiseg.utils.alienlab as alien

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



Sx = param2['Sx']
Sy = param2['Sy']

epochs = param2['epochs']
batch_size = param2['batch']

learning_rate = param2['learning_rate']


############################################################################################################################

def cnn_eval(directory_weights, directory_dataset, tsboard, batch_size, epochs,
                    model, Sx, Sy, load_model = False):
        
    #Training board
    
    #image transformation for training, can be modified for data augmentation
    trans = transforms.Compose([
                                transforms.CenterCrop((Sx, Sy)),
                                transforms.ToTensor(),
                                ])
    
    #Load images and ground truth
    path_test = directory_dataset
    shots, channels = init_set('', path_test)

    test_dataset = Dataset_im_label(shots, channels, transform = trans, path = path_test)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
   # with torch.no_grad():
    if True:
        ared_tot = []
        id_list = []
        count = 0
        print('Image segmentation by the CNN')
    
        for inputs, id_im in test_loader:
            inputs = inputs.to(device) #input image on GPU
            outputs = evaluate(inputs, model)  #output image
            
            pred_tot.append(outputs)
            del outputs
            del model
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
       
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = torch.load(directory_weights + '/' + model_segmentation_name)
    """
    train_bce = model[1]['bce'][::2]
    val_bce = model[1]['bce'][1::2]
    
    train_dice = model[1]['dice'][::2]
    val_dice = model[1]['dice'][1::2]
    
    plt.plot(train_bce, 'training bce error')
    plt.plot(val_bce, 'validation bce error')
    
    plt.plot(train_dice, 'training dice error')
    plt.plot(val_dice, 'validation dice error')
    
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.show()
    """
    
    trans = transforms.Compose([
                                transforms.CenterCrop((Sx, Sy)),
                                transforms.ToTensor(),
                                ])
    
    #Load images and ground truth
    path_test = directory_dataset +'/test/'
    shots, channels = init_set('', path_test)

    test_dataset = Dataset_im_label(shots, channels, transform = trans, path = path_test ) 
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    with torch.no_grad():
    
        pred_tot = []
        id_list = []
        count = 0
        loss_tot = {'bce':0, 'dice':0} 
        print('Image segmentation by the CNN')
        im_class = [] 
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device) #input image on GPU
            labels = labels.to(device)

            outputs, metrics = test(inputs, labels, model[0].eval().to(device))  #output image

            
            torch.cuda.empty_cache()
            pred_tot.append(inputs[0].permute(1,2,0).cpu())
            im_class.append('image')
            
            ind_class = random.choice([0,1,2,3,4])
            pred_tot.append(outputs[0, ind_class].cpu())
            im_class.append(channels[ind_class])

            loss_tot['bce'] += metrics['bce']
            loss_tot['dice'] += metrics['dice']
            count += 1
           
            
        loss_tot['dice'] /= count
        loss_tot['bce'] /= count
    print("Loss over test set: ", loss_tot)
    g = alien.showclass()
    g.save_name = model_segmentation_name[:-3]
    g.figsize=(100,100)
    g.col_num = 5
    g.save_folder = ''
    g.date = True
    g.title_list = im_class
    g.multi(pred_tot[0:25])
    #g.showing()
