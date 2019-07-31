#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:27:45 2019

@author: alienor
"""

import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import glob
import random

#%%

path_imgs = "data/arabidopsis/"


trans = transforms.Compose([
        #transforms.Resize((896,448)),
        transforms.CenterCrop((896, 448)),
        #transforms.Resize(224),
        # you can add other transformations in this list
        transforms.ToTensor()])

def init_set(mode, path = path_imgs, ext = "png"):
    if mode not in ['train', 'val', 'test']:
    
        print("mode should be 'train', 'val' or 'test'")
    image_paths = glob.glob(path + mode + "/images/*." + ext)
    target_paths = glob.glob(path + mode + "/labels/*.png")
    return np.sort(image_paths), np.sort(target_paths)




class CustomDataset(Dataset):

    def __init__(self, image_paths, target_paths, transform = trans):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transform

    def __getitem__(self, index):
        angle = random.randint(0,360)
        image = Image.open(self.image_paths[index])
        image = image.rotate(angle)
        mask = Image.open(self.target_paths[index])
        mask = mask.rotate(angle)
        t_image = self.transforms(image)
        #t_image = transforms.Normalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.22])(t_image)

        t_mask = self.transforms(mask)
        
        t_mask = self.read_label(t_mask, self.target_paths[index])
        
        #t_mask = t_mask.permute(1,2,0)
        t_image = t_image[0:3, :, :]
        return t_image, t_mask

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)
    
    def read_label(self, im, name):
        '''This function reads the binary-encoded label of the input image and
        returns the one hot encoded label. 6 classes: 5 plan organs and ground'''
        im1 = (im[0]*255).type(torch.int32)
        a, b = im1.shape
        label_image = torch.zeros((6, a, b))
        if 'plant' not in name:
            im = (im[0]*255).type(torch.int32)
            label_image[0][im == 0] = 1

            for i in range(1,6): #[binary reading]
                label_image[i] = im%2
                im = im//2
        else:
            im = (im[2]*255).type(torch.int32)
            label_image[0][im == 0] = 1
            for i in range(1, 6):
                label_image[i][im//51 == i] = 1
                
        return label_image

if False: 
    image_paths, target_paths = init_set('train', path = path_imgs)
    
    train_dataset = CustomDataset(image_paths, target_paths, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)


if False:
    for batch_idx, (inputs, labels) in enumerate(train_loader):
         print(labels.shape)
         inputs = labels.permute(2, 3, 0, 1)
         plt.figure()
         plt.imshow(inputs[:,:,0,0].numpy())
         plt.show()
  
#%%
         
       
class Dataset_3D(Dataset):

    def __init__(self, image_paths, target_paths, transform = trans, the_shape=2197000):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transform
        self.the_shape = the_shape

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        mask = torch.load(self.target_paths[index])
        t_image = self.transforms(image)
        
        t_mask = torch.sparse.FloatTensor(mask[0].unsqueeze(0), mask[1], 
                              size = torch.Size([self.the_shape])).to_dense()
        
        #t_mask = t_mask.permute(1,2,0)
        t_image = t_image[0:3, :, :]
        return t_image, t_mask

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)
    
def init_3D_set(mode, path = path_imgs, N_subsample = 1, N_cam = 72):
    if mode not in ['train', 'val', 'test']:
    
        print("mode should be 'train', 'val' or 'test'")
        
    image_paths = np.sort(glob.glob(path + mode + "/images/*.jpg"))
    target_paths = np.sort(glob.glob(path + mode + "/3D_label/*.pt"))
    target_paths = np.repeat(target_paths, N_cam)

    return  image_paths[::N_subsample], target_paths[::N_subsample]


trans = transforms.Compose([
    transforms.CenterCrop((896, 448)),
    transforms.ToTensor()])