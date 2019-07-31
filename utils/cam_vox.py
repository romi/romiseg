#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:31:55 2019

@author: alienor
"""

import torch
import torch.nn as nn
from torchvision import models
from collections import defaultdict
import torch.nn.functional as F
from utils.dataloader import Dataset_3D
import utils.alienlab as alien
import numpy as np
import glob
from utils.ply import write_ply
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import utils.vox_to_coord as vtc
import matplotlib.pyplot as plt 
import json
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
#%%

labels_names = ['background', 'flowers', 'peduncle', 'stem', 'leaves', 'fruits']

#path_json = 'data/annotations/test/camera/'


def get_cam_param(path_json, NP, scale = 1, N_cam = 72):
    #with open(path_json + 'images_plant%d.json'%NP, 'r') as f:
    with open(path_json + 'images.json', 'r') as f:
        pose = json.load(f)

    extrinsics = torch.zeros((N_cam, 3, 4))
    for i in range(N_cam):
        rot = pose[str(i+1)]['rotmat']
        extrinsics[i][:3,:3] = torch.Tensor(rot)#.transpose(0, 1)
        trans = pose[str(i+1)]['tvec']
        extrinsics[i][:,3] = torch.Tensor(trans)
       
   # with open(path_json + 'cameras_plant%d.json'%NP, 'r') as f:
    with open(path_json + 'cameras.json', 'r') as f:
        focal = json.load(f)
    focal = focal['1']['params']
    
    r = 1/scale
    
    intrinsics = torch.zeros((1, 3, 3))
    intrinsics[:,0,0] = focal[0]*r
    
    intrinsics[:,1,1] = focal[0]*r
    intrinsics[:,0,2] = focal[1]*r
    intrinsics[:,1,2] = focal[2]*r
    intrinsics[:,2,2] = 1
    #print(extrinsics[0], intrinsics)
    
    m = (-extrinsics[:,:,:3].permute(0, 2, 1) @ extrinsics[:,:,3].unsqueeze(-1)).mean(dim = 0)
    
    return extrinsics, intrinsics, m

def build_voxel_volume(extrinsics, intrinsics, min_vox, num_vox = 100, N_cam  = 72, cloud_scale = 2,
                       Sx= 896, Sy = 448, xinit = 1080, yinit = 1616, label_num = 6):
    #Voxel representation of the point cloud

    basis_voxels = vtc.basis_vox(min_vox, num_vox[0], num_vox[1], num_vox[2])*cloud_scale#List of coordinates  
    
    v = cloud_scale//2
    basis_voxels[:,0] += v
    inds = [2,2,1,2,0,1,2,1]
    val = [1, -1, 1, 1, -1, -1, -1, 1]
    
        
    for i in range(len(inds)):
        basis_voxels[:, inds[i]] += v * val[i]
        print(basis_voxels[0])
        #Camera projection
        torch_voxels = torch.from_numpy(basis_voxels)    
       
        #Perspective projection
        xy_coords = vtc.project_coordinates(torch_voxels, intrinsics, extrinsics, give_prod = False)
        
        #permute x and y coordinates
        xy_coords[:, 2, :] = xy_coords[:,0,:]
        xy_coords[:, 0, :] = xy_coords[:,1,:]
        xy_coords[:, 1, :] = xy_coords[:,2,:]
        
        coords = vtc.correct_coords_outside(xy_coords, Sx, Sy, xinit, yinit, -1) #correct the coordinates that project outside
        the_shape = torch.Size([N_cam, xinit, yinit, label_num])
        xy_full_flat = vtc.flatten_coordinates(coords, the_shape)
        
        torch.save(torch_voxels, 'voxel_coord/voxels_real_%d.pt'%i)
        torch.save(xy_full_flat, 'voxel_coord/coordinates_real_%d.pt'%i)
        
        write_ply('cage_%d.ply'%i, torch_voxels.detach().cpu().numpy(),
          ['x', 'y', 'z', 'labels'])
        del torch_voxels
        del xy_full_flat