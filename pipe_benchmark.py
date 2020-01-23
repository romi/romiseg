#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:18:24 2019

@author: alienor
"""
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
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn

#from torchvision import models

from romiseg.utils.train_3D import train_model_voxels, init_set, Dataset_im_label_3D
from romiseg.utils.train_from_dataset import plot_dataset
from romiseg.utils import segmentation_model

import romiseg.utils.vox_to_coord as vtc
from romiseg.utils.generate_volume import generate_volume


default_config_dir = "/home/alienor/Documents/scanner-meta-repository/Scan3D/config/segmentation2d_guitar.toml"

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--config', dest='config', default=default_config_dir,
                    help='config dir, default: %s'%default_config_dir)


args = parser.parse_args()



param_pipe = toml.load(args.config)

direc = param_pipe['TrainingDirectory']

path = direc['path']
directory_weights = path + direc['directory_weights']
tsboard = path +  direc['tsboard'] + '/full_pipe'
directory_dataset = path + direc['directory_dataset']


param2 = param_pipe['Segmentation2D']

label_names = param2['labels'].split(',')


Sx = param2['Sx']
Sy = param2['Sy']

epochs = param2['epochs']
batch_size = param2['batch']

learning_rate = param2['learning_rate']
model_name = param2['model_name']


param3 = param_pipe['Reconstruction3D']
N_vox = param3['N_vox']
coord_file_loc = path + param3['coord_file_loc']



############################################################################################################################




generate_volume(directory_dataset + '/train/', coord_file_loc, Sx, Sy, N_vox, label_names)
generate_volume(directory_dataset + '/val/', coord_file_loc, Sx, Sy, N_vox, label_names)

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
path_val = directory_dataset + '/val/'
path_train = directory_dataset + '/train/'

image_train, target_train, voxel_train = init_set('', path_train)
image_val, target_val, voxel_val = init_set('', path_val)


    
train_dataset = Dataset_im_label_3D(image_train, target_train, voxel_train, transform = trans)
val_dataset = Dataset_im_label_3D(image_val, target_val, voxel_val, transform = trans) 
        
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

'''
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
   
voxels = torch.load(coord_file_loc + '/voxels.pt').to(device)
      
model = segmentation_model.ResNetUNet_3D(num_classes, coord_file_loc).to(device)


# freeze backbone layers
for l in model.base_layers:
    for param in l.parameters():
        param.requires_grad = False
   
#Choice of optimizer, can be changed
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
#make learning rate evolve
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

#Run training
w_back = 1
w_class = 30
weights = [w_back] + [w_class]*(num_classes-1) #[ 1 / number of instances for each class]
class_weights = torch.FloatTensor(weights).cuda()




voxel_loss = nn.CrossEntropyLoss(weight=class_weights)

ext_name = '_segmentation_' + str(Sx) + '_' + str(Sy) + '_epoch%d.pt'%epochs
new_model_name = model_name + ext_name

if True:
    model = train_model_voxels('Segmentation', dataloaders, model, optimizer_ft, exp_lr_scheduler, writer, voxel_loss, voxels,
                        num_epochs = epochs, viz = True, label_names = label_names)
        
    #model[0].save_state_dict(directory_weights + '/' + new_model_name)
    torch.save(model, directory_weights + '/' + new_model_name)
    model = model[0]
else:
    model = torch.load(directory_weights + '/' + new_model_name)[0].to(device)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    'test' : DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate*0.1)


model.class_layer[0].weight.data.fill_(0)
model.class_layer[0].weight.data.fill_diagonal_(1)
model.class_layer[0].bias.data.fill_(0)
print(model.class_layer[0].weight.data)

model = train_model_voxels('Fullpipe', dataloaders, model, optimizer_ft, exp_lr_scheduler,
                           writer, voxel_loss, voxels,
                    num_epochs = epochs, viz = True, label_names = label_names)

#save model
model_segmentation_name =  new_model_name + os.path.split(directory_dataset)[1] + '_epoch%d.pt'%epochs
torch.save(model, directory_weights + '/' + model_segmentation_name)

'''
    return model, model_name

#######
cnn_train(directory_weights, directory_dataset, label_names, tsboard, batch_size, epochs,
                    model_segmentation_name, Sx, Sy)
'''
model = torch.load(directory_weights + '/' + model_segmentation_name)[0].to(device)
accuracy = []


for image, label, voxel in dataloaders['train']:
    image = image.to(device)
    pred_im, pred_vox = model(image)
    voxel = voxel[0,:,3].unsqueeze(1).long()
    onehot = torch.zeros((voxel.shape[0],4))
    onehot = onehot.scatter_(1, voxel, 1)
    accuracy.append(torch.sum(onehot*pred_vox.cpu())/voxel.shape[0])
    del image, label, voxel, onehot, pred_im, pred_vox
    print(accuracy)
    
print(np.mean(accuracy))
