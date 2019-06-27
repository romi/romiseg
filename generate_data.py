#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:08:39 2019

@author: alienor
"""
#path = '/home/alienor/Documents/blender_virtual_scanner'
#import sys
#sys.path.append(path)
from utils.save_images import virtual_scan 
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os 
from utils.ply import *
import utils.generate_3D_ground_truth as gt_vox

import utils.proj_point_cloud as ppc
import utils.vox_to_coord as vtc
import matplotlib.pyplot as plt
import utils.tests as test

import json
import torch
#%%
if __name__ == "__main__":
    W = 1920
    H = 1080


#BLENDER IMAGES
if False: 
    for i in range(0, 50):

        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'train')
       
    for i in range(50, 80):
        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'val')
    
    for i in range(80, 100):
        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'test')
#%%
        
if False: 
    W = 1920//2
    H = 1080//2
    
    for i in range(0, 50):

        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'train_2')
       
    for i in range(50, 80):
        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'val_2')
    
    for i in range(80, 100):
        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'test_2')
#%%
#CHANGE MATERIALS
NN = 100
if False: 
    for i in range(NN):
        openFile=os.system("blender -b -P utils/change_materials.py -- %d"%i)
        
#%%
#MESH TO POINT CLOUD
if False:
    for n in range(NN):
        directory = "LEARNING/ground_truth_3D/"
        filepath = directory + "arabidopsis_3D_GT_%03d"%n
        filename = filepath + ".obj"
        savename = filepath + ".ply"
        
        density = 100 
        
        openFile=os.system("CloudCompare -SILENT -O " + filename + 
                       " -C_EXPORT_FMT PLY -AUTO_SAVE OFF -SAMPLE_MESH DENSITY %d -SAVE_CLOUDS FILE "%density + savename)

        
        gt_vox.gt_pc_to_vox(savename)
        
#%% Image and 3D Ground truth
        
if True: 
    directory = "LEARNING/ground_truth_3D/"
    #filename = directory +  name + ".obj"
    #savename = "data/arabidopsis/" + name + ".ply"
    
    
    for i in range(0, 50):
        
        locname = directory + "arabidopsis_3D_GT_%03d.ply"%i
        #filename = filepath + ".obj"
        name = "arabidopsis_3D_GT_%03d"%i
        savename = "data/arabidopsis/train/3D_label/" + name
    
        #openFile=os.system("CloudCompare -SILENT -O " + filename + 
        #               " -C_EXPORT_FMT PLY -AUTO_SAVE OFF -SAMPLE_MESH DENSITY %d -SAVE_CLOUDS FILE "%density + savename)
        
        v = gt_vox.gt_pc_to_vox(locname, savename)
        print(v)
        
        #tool = virtual_scan(w = W, h = H)
        #tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'train')
       
    for i in range(50, 80):
        #tool = virtual_scan(w = W, h = H)
        #tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'val')
        locname = directory + "arabidopsis_3D_GT_%03d.ply"%i
        name = "arabidopsis_3D_GT_%03d"%i

        #filename = filepath + ".obj"
        savename = "data/arabidopsis/val/3D_label/" + name
    
        #openFile=os.system("CloudCompare -SILENT -O " + filename + 
        #               " -C_EXPORT_FMT PLY -AUTO_SAVE OFF -SAMPLE_MESH DENSITY %d -SAVE_CLOUDS FILE "%density + savename)
        
        gt_vox.gt_pc_to_vox(locname, savename)
        
    
    for i in range(80, 100):
        #tool = virtual_scan(w = W, h = H)
        #tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'test')
        locname = directory + "arabidopsis_3D_GT_%03d.ply"%i
        name = "arabidopsis_3D_GT_%03d"%i

        #filename = filepath + ".obj"
        savename = "data/arabidopsis/test/3D_label/" + name
    
        #openFile=os.system("CloudCompare -SILENT -O " + filename + 
        #               " -C_EXPORT_FMT PLY -AUTO_SAVE OFF -SAMPLE_MESH DENSITY %d -SAVE_CLOUDS FILE "%density + savename)
        
        gt_vox.gt_pc_to_vox(locname, savename)
   
     

#%%
#voxel and coordinates shifts (initial no shift: shift6)
if False:
    with open('images.json', 'r') as f:
        pose = json.load(f)

    N_cam = 72
    N_feat = 72
    red_factor = 1
    
    extrinsics = torch.zeros((N_cam, 3, 4))
    for i in range(N_cam):
        rot = pose[str(i+1)]['rotmat']
        extrinsics[i][:3,:3] = torch.Tensor(rot)
        trans = pose[str(i+1)]['tvec']
        extrinsics[i][:,3] = torch.Tensor(trans)/10.
    
    with open('cameras.json', 'r') as f:
        focal = json.load(f)
    focal = focal['1']['params']
    
    r = 1/red_factor
    
    intrinsics = torch.zeros((1, 3, 3))
    intrinsics[:,0,0] = focal[0]*r
    intrinsics[:,1,1] = focal[0]*r
    intrinsics[:,0,2] = focal[1]*r
    intrinsics[:,1,2] = focal[2]*r
    intrinsics[:,2,2] = 1
    
    cloud_scale = 0.5
    sc = 5
    v= 0.25
    Sx = 896 #Center crop
    Sy = 448
    xinit = 1080 #Original image size
    yinit = 1920
    N_class = 6
    the_shape = torch.Size([N_cam, xinit, yinit, N_class])
    

    N = int(65/cloud_scale)
    #Voxel representation of the point cloud
    min_vec = [int(-40/cloud_scale), int(-40/cloud_scale),int(-5/cloud_scale)] #Limit of the cloud
    basis_voxels = vtc.basis_vox(min_vec, N, N, N)*cloud_scale#List of coordinates 
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

        xy_full_flat = vtc.flatten_coordinates(coords, the_shape)
        torch.save(xy_full_flat, 'voxel_coord/coordinates_05_shift%d.pt'%i)
        torch.save(torch_voxels, 'voxel_coord/voxels_05_shift%d.pt'%i)
        del torch_voxels
        del xy_full_flat
        

    