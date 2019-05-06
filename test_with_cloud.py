#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:00:46 2019

@author: alienor
"""
import os#, subprocess

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply
from utils.vox_to_coord import *
from utils.proj_point_cloud import *
from utils.tests import *

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
import imageio
#%%
if __name__ == "__main__":

    
    #FROM A POINT CLOUD
    
    n = 4
    filepath = r"data/arabidopsis_%d"%n
    density = 10 #pts/cm³
    cloud_scale = 0.5
    
    
    N = int(65./cloud_scale)
    #Voxel representation of the point cloud
    min_vec = [int(-40/cloud_scale), int(-40/cloud_scale),int(-5/cloud_scale)] #Limit of the cloud
    basis_voxels = basis_vox(min_vec, N, N, 3*N) #List of coordinates  

    #Load mesh as point cloud by calling cloudcompare
    points = load_mesh_as_pcd(filepath, n, density, cloud_scale)
#%% 
    voxels = fill_vox(points, basis_voxels, cloud_scale, min_vec, N, N, 3*N) #Fill the coordinates
    #test
    plot_3D(voxels) #Check the 3D point cloud representation in the voxels
 
#%%   
    #Camera projection

    N_cam = 72
    x0, y0, z0 = 100./cloud_scale, 0., -100./cloud_scale
    rx, ry = 0, 0
    fx = 24
    fy = 24
    cx = 0
    cy = 0
    x = x0
    y = y0
    z = z0
    #All the camera matrices for each position
    intrinsics = torch.tensor([[fx, 0], [0,fy]])
    intrinsics = intrinsics.unsqueeze(0)
 
    extrinsics = get_trajectory(N_cam, x0, y0, z0, rx, ry)
 
    torch_voxels = torch.from_numpy(voxels)

    #Perspective projection
    prod, xy_coords = project_coordinates(torch_voxels, intrinsics, extrinsics)

    ind = torch_voxels[:,3]==1
    #test
    #move_camera(torch_voxels, extrinsics, N_cam, prod)  
    #local_proj(xy_coords, N_cam)  
    s = 32./1920.          
    maxi = torch.tensor([1920*s*cloud_scale, 1080*s*cloud_scale])
  
    
    #View the images 

    images = views_to_images(torch_voxels, xy_coords, s, cloud_scale, N_cam, maxi)
    imageio.mimsave('test_cloud/virtual_scan_torch%d.gif'%n, images)
