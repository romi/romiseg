#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:53:46 2019

@author: alienor
"""

import os#, subprocess

# Import functions to read and write ply files
from romiseg.utils.ply import write_ply, read_ply
import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
import imageio

def avoid_eps(a, eps):
    a[torch.abs(a)<eps] = 0
    return a


def basis_vox(min_vec, w, h, l):
    """
    Voxelize a point cloud. List of voxel coordinates occupied by a point.
    """

    minx, miny, minz = min_vec
    X = np.linspace(minx, minx + w -1, w)
    Y = np.linspace(miny, miny + w -1, h)
    Z = np.linspace(minz, minz + w -1, l)
    
    voxels = np.zeros((w*h*l,4))
    voxels[:,0] = np.repeat(X, h*l)
    Y_bis = np.repeat(Y, l)
    voxels[:,1] = np.tile(Y_bis, w)
    voxels[:,2] = np.tile(Z, h*w)
    
    return voxels

def basis_vox_pipeline(min_vec, max_vec, w, h, l):
    """
    Voxelize a point cloud. List of voxel coordinates occupied by a point.
    """

    minx, miny, minz = min_vec
    maxx, maxy, maxz = max_vec
    X = np.linspace(minx, maxx, w)
    Y = np.linspace(miny, maxy, h)
    Z = np.linspace(minz, maxz, l)
    
    voxels = np.zeros((w*h*l,4))
    voxels[:,0] = np.repeat(X, h*l)
    Y_bis = np.repeat(Y, l)
    voxels[:,1] = np.tile(Y_bis, w)
    voxels[:,2] = np.tile(Z, h*w)
    
    return voxels

def get_int(fx, fy, cx, cy):
    intri = torch.tensor([[fx, 0, cx],[0,fy,cy],[0,0,1]])
    return intri

def get_extr(rx, ry, rz, x, y, z):
    rt = torch.zeros((3,4))

    rotx = torch.zeros((3,3))
    c = np.cos(rx)
    s = np.sin(rx)
    rotx[0,0] = 1
    rotx[2,2] = c
    rotx[1,2] = -s
    rotx[2,1] = s
    rotx[1,1] = c
        
    roty = torch.zeros((3,3))
    c = np.cos(ry)
    s = np.sin(ry)
    roty[0,0] = c
    roty[2,2] = c
    roty[0,2] = s
    roty[2,0] = -s
    roty[1,1] = 1
    
    rotz = torch.zeros((3,3))
    c = np.cos(rz)
    s = np.sin(rz)
    rotz[0,0] = c
    rotz[1,1] = c
    rotz[0,1] = -s
    rotz[1,0] = s
    rotz[2,2] = 1
    
    eps = 1e-7
    rotz = avoid_eps(rotz, eps)
    roty = avoid_eps(roty, eps)
    rotx = avoid_eps(rotx, eps)

    print(rotx)
    print(roty)
    print(rotz)

    rot = torch.mm(torch.mm(rotx, roty), rotz)
    trans = torch.tensor([[x], [y], [z]])
    trans = -torch.mm(torch.transpose(rotz, 0, 1), trans)
    rt[:3,:3] = rot
    rt[:,3] = torch.transpose(trans, 0, 1)
    
    return rt


def get_trajectory(N_cam, x0, y0, z0, rx, ry):
    d_theta = -2 * np.pi/N_cam
    extrinsics = torch.zeros(N_cam, 3, 4)
    for i in range(N_cam):
        x1 = x0 * np.cos(i * d_theta) #x pos of camera
        y1 = x0 * np.sin(i * d_theta) #y pos of camera 
        rz = d_theta * i + np.pi/2 #camera pan   
        pose = get_extr(rx, ry, rz, x1, y1, z0)
        extrinsics[i, :,:] = pose
    return extrinsics

def project_coordinates(torch_voxels, intrinsics, extrinsics, give_prod):
    torch_voxels = torch_voxels.unsqueeze(0) #homogeneous coordinates
    t = torch_voxels.clone()
    #t = t-torch.mean(t, dim = 1)
    t[:,:,3]=1

    t = t.permute(0, 2, 1) #convenient for matrix product
    ext = extrinsics[:,0:3,:] #several camera poses
    prod = torch.matmul(ext.double(), t) #coordinate change
    if give_prod == True:
        prod1 = prod.clone()
    prod[:,0,:] = prod[:,0,:]/prod[:,2,:]
    prod[:,1,:] = prod[:,1,:]/prod[:,2,:]
    prod[:,2,:] = prod[:,2,:]/prod[:,2,:]
    xy_coords = prod[:, 0:3, :]
    xy_coords = torch.matmul(intrinsics.double(), xy_coords) #coordinate change
    if give_prod == True:
        return prod1, xy_coords
    else: 
        return xy_coords

def correct_coords_outside(coordinates, Sx, Sy, xinit, yinit, val):
    '''This function ensures that the voxels ot the volume that 
    don't project onto the different views won't generate index errors when we project them onto the predictions
    We will associate to them a specific label meaning "voxel projected outside of the image"
    Inputs: -coodinates xy for each view (torch tensor)
            -center crop dimensions
            -image dimensions
            -value to assign to the coordinates projectiong outside (-1 is a good choice)
    Output: Modified coordinates
    '''
    
    coords = coordinates.clone()
    
    indices = (coords[:,0,:] < (xinit-Sx)/2) #lower bound for x
    ind_stack = torch.stack([indices]*3, dim = 1)
    coords[ind_stack] = val
    
    indices = (coords[:,1,:] < (yinit-Sy)/2) #lower bound for y
    ind_stack = torch.stack([indices]*3, dim = 1)
    coords[ind_stack] = val
    
    indices = (coords[:,0,:] > (xinit+Sx)/2) #upper bound for x
    ind_stack = torch.stack([indices]*3, dim = 1)
    coords[ind_stack] = val
    
    indices = (coords[:,1,:] > (yinit+Sy)/2) #upper bound for y
    ind_stack = torch.stack([indices]*3, dim = 1)
    coords[ind_stack] = val
    
    return coords.long()


def adjust_predictions(preds):
    '''
    The predictions have to be flattened to be accessed by the coordinates tensor. 
    Also a new class has to be added to 
    describe the voxels prohjecting outside the image. 
    These voxels will project onto the last element of the flattened predictions
    that correspond to "pixel outside the image" class.
    Input: predictions in shape (N_cam, W, H, num_labels) torch tensor
    Output: Flattened predictions (N_cam * W * H + 1, num_labels + 1)
    '''
    outside_proj_label = (preds[:,:,:,0]*0).unsqueeze(-1) 
    preds = torch.cat([preds, outside_proj_label], dim = 3) #Add a label class: voxel projects outside image
    preds_flat = torch.flatten(preds, end_dim = -2) #Flatten the predictions
    
    outside_label = preds_flat[0] * 0
    outside_label[-1] = 1 
    outside_label = outside_label.unsqueeze(0)
    preds_flat = torch.cat([preds_flat, outside_label]) #Add a last prediction where all 
    #voxels that project outside the  image will collect their class
    
    return preds_flat


def flatten_coordinates(coords, shape_predictions):
    '''
    To access the predictions by indexing, the indexes have to be modified according to the 
    flattened predicitions.
    Inputs: -xy coordinates
            -shape of the predictions before flattening
    Output: adapted coordinates + take into account the coordinates that project outside the images
            they will project onto the last element of the flattened predictions
    '''
    xx = coords[:,0]
    yy = coords[:,1]
    xy = torch.mul(xx, shape_predictions[2]) + yy  #manually perform ravel_multi_index from numpy, along X and Y 
    
    flat_factor = shape_predictions[1] * shape_predictions[2] 
    flat_vec = torch.mul(torch.linspace(0, shape_predictions[0] - torch.tensor(1), shape_predictions[0]), flat_factor)
    flat_vec = flat_vec.unsqueeze(1).long()
    flat_coo = torch.add(xy, flat_vec) #Perform it along the views N_cam
    flat_coo[xy < 0] = -1 #Set the negative indexes 

    xy_full_flat = torch.flatten(flat_coo)
    
    
    return xy_full_flat