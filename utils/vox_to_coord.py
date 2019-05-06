#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:53:46 2019

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


def basis_vox(min_vec, w, h, l):
    """
    Voxelize a point cloud. List of voxel coordinates occupied by a point.
    """

    minx, miny, minz = min_vec
    X = np.linspace(minx, minx + w-1, w)
    Y = np.linspace(miny, miny + h-1, h)
    Z = np.linspace(minz, minz + l-1, l)
    
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
        
    rot = torch.mm(rotz, torch.mm(rotx, roty))
    trans = torch.tensor([[x], [y], [z]])
    #trans = -torch.mm(torch.transpose(rot, 0, 1), trans)
    rt[:3,:3] = rot
    rt[:,3] = torch.transpose(trans, 0, 1)
    
    return rt


def get_trajectory(N_cam, x0, y0, z0, rx, ry):
    d_theta = 2 * np.pi/N_cam
    extrinsics = torch.zeros(N_cam, 3, 4)
    for i in range(N_cam):
        rz = d_theta * i + np.pi #camera pan      
        pose = get_extr(rx, ry, rz, x0, y0, z0)
        extrinsics[i, :,:] = pose
    return extrinsics

def project_coordinates(torch_voxels, intrinsics, extrinsics):
    torch_voxels = torch_voxels.unsqueeze(0) #homogeneous coordinates
    t = torch_voxels.clone()
    #t = t-torch.mean(t, dim = 1)
    t[:,:,3]=1

    t = t.permute(0, 2, 1) #convenient for matrix product
    ext = extrinsics[:,0:3,:] #several camera poses
    prod = torch.matmul(ext.double(), t) #coordinate change
    prod[:,1,:] /= prod[:,0,:]
    prod[:,2,:] /= prod[:,0,:]
    xy_coords = prod[:, 1:3, :]
    xy_coords = torch.matmul(intrinsics.double(), xy_coords) #coordinate change

    return prod, xy_coords

