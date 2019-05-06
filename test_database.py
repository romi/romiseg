#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:14:29 2019

@author: alienor
"""


from utils.dataloader import init_set, CustomDataset
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

if __name__ == "__main__":
    
    B = 4

    image_paths, target_paths = init_set('train')
    
    train_dataset = CustomDataset(image_paths, target_paths, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=1)
    


    images, label = next(iter(train_loader))
    
    #plot 4 images to visualize the data
    rows = int(np.sqrt(B))
    columns = int(np.sqrt(B))*2
    fig=plt.figure(figsize = (10, 4))
    for i in range(B):
       fig.add_subplot(rows, columns, 2*i+1)
       plt.axis('off')
       plt.grid(False)
       img = images[i]
       img = torchvision.transforms.ToPILImage()(img)
       plt.imshow(img)
       fig.add_subplot(rows, columns, 2*i+2)
       plt.axis('off')
       plt.grid(False)
       img = label[i,3,:,:].int()
       img = torchvision.transforms.ToPILImage()(img)
       plt.imshow(img)
    plt.show()
    
