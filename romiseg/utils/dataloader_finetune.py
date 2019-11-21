#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:54:46 2019

@author: alienor
"""



import torch
from torch.utils.data import Dataset

import glob
import numpy as np
from PIL import Image
import random

import romiseg.utils.alienlab as alien



def init_set(mode, path, ext = "png"):

    image_paths = glob.glob(path + mode + "/images/*.jpg") + glob.glob(path + mode + "/images/*.png")
    target_paths = glob.glob(path + mode + "/labels/*.jpg") + glob.glob(path + mode + "/labels/*.png")
    return np.sort(image_paths), np.sort(target_paths)




class Dataset_im_label(Dataset):

    def __init__(self, image_paths, target_paths, transform, rotate_bool = True):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transform
        self.rotate_bool = rotate_bool

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        if self.rotate_bool == True:     
            angle = random.randint(0,360)
            image = image.rotate(angle)
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
        
        '''
        im = (im[2]*255).type(torch.int32)
        label_image[0][im == 0] = 1
        for i in range(1, 6):
            label_image[i][im//51 == i] = 1
        '''
        ind = im.shape[0]-1
        im = (im[ind]*255).type(torch.int32)
        label_image[0][im == 0] = 1

        for i in range(1,6): #[binary reading]
            label_image[i] = im%2
            im = im//2        
        return label_image
    

def plot_dataset(train_loader, labels_names, batch_size, showit = True):
    images, label = next(iter(train_loader))
    
    #plot 4 images to visualize the data
    images_tot = []
    titles_tot = []
    for i in range(batch_size):
        img = images[i]
        img = img.permute(1, 2, 0)
        images_tot.append(img)
        titles_tot.append('image')
        img = label[i,i,:,:].int()
        images_tot.append(img)
        titles_tot.append(labels_names[i])
    g = alien.showclass()
    g.save_im = False

    g.col_num = 3
    g.figsize = ((14, 14))
    g.title_list = titles_tot
    fig = g.showing(images_tot, showit,)
    return fig
