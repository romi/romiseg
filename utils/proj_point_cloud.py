#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:57:39 2019

@author: alienor
"""
import os#, subprocess

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply
import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
import imageio

def load_mesh_as_pcd(filepath, n, density, cloud_scale):

    filename = filepath + ".obj"
    savename = filepath + ".ply"
    openFile=os.system("CloudCompare -SILENT -O " + filename + 
                   " -C_EXPORT_FMT PLY -AUTO_SAVE OFF -SAMPLE_MESH DENSITY %d -SAVE_CLOUDS FILE "%density + savename)
    data = read_ply(savename)
    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    #points = points - np.mean(points, axis = 0)
    #points[:,0] = points[:,0] - np.mean(points[:,0])
    #points[:,1] = points[:,1] - np.mean(points[:,1])
    return points

def fill_vox(point_cloud, basis_voxels, vox_size, min_vec, w, h, l):
    cloud = np.copy(point_cloud)
    cloud[:,0] = cloud[:,0]
    cloud[:,1] = cloud[:,1]
    cloud[:,2] = cloud[:,2]
    voxels = cloud // vox_size

    
    for coords in voxels:
        ind = int(np.array([h*l, l, 1]).dot(coords - min_vec))
        basis_voxels[ind, 3] = 1
        
    return basis_voxels


def views_to_images(torch_voxels, xy_coords, s, cloud_scale, N_cam, maxi):
    ind = torch_voxels[:,3]==1
    coords = xy_coords[:,:,ind].clone()
    maxi = maxi/cloud_scale
    
    #coords[:, 2, :] = coords[:,0,:]
    #coords[:, 0, :] = coords[:,1,:]
    #coords[:, 1, :] = coords[:,2,:]

    coords[:,0] = coords[:,0]# - torch.mean(coords[:,0]) + maxi[0]/2
    print(torch.mean(coords[:,0]), torch.max(coords[:,0]),torch.min(coords[:,0]))
    coords[:,1] = coords[:,1]# - torch.mean(coords[:,1]) + maxi[1]*1/2
    print(torch.mean(coords[:,1]), torch.max(coords[:,1]),torch.min(coords[:,1]))

    maxi = maxi/s
    image = torch.zeros(int(maxi[0])+1, int(maxi[1])+1)
  
    #coords = coords/s
    coords = coords.int()
    images = []
    for i in range(N_cam):
        image = torch.zeros(int(maxi[0])+1, int(maxi[1])+1)
        ind_i = coords[i][0,:]>0
        print(np.count_nonzero(ind_i))
        coords_i = coords[i][:, ind_i]
        ind_i = coords_i[1,:]>0
        print(np.count_nonzero(ind_i))
        coords_i = coords_i[:, ind_i]
        ind_i = coords_i[0,:] < int(maxi[0])
        print(torch.max(coords_i[0,:]))
        coords_i = coords_i[:, ind_i]
        ind_i = coords_i[1,:] < int(maxi[1])
        print(torch.max(coords_i[1,:]))
        coords_i = coords_i[:, ind_i]

        image[coords_i[0,:].long(), coords_i[1,:].long()] = 1
        #image = image.permute(1,0)

        images.append(image)
        
        
    return images