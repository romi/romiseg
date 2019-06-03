#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:00:01 2019

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

def plot_3D(voxels):
    '''
    Input [np array 4xN³]: 4 columns, XYZ and 0 or 1, rows: number of points in 
    the 3D cube that contains the point cloud.
    Output: 3D figure
    Plots a 3D point cloud from a list of coordinates xyz corresponding to 3D coordinates
    of a 3D cube that contains a point cloud. For the coordinates corresponding to a point
    the cloud, the element of the last column is 1, else it's 0.
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vox = voxels[voxels[:,3]>1]
    ax.scatter(vox[:,0], vox[:,1], vox[:,2], s = 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_aspect('equal')
    plt.show()
    
def move_camera(torch_voxels, extrinsics, N_cam, prod):
    ind = torch_voxels[:,3]==1
    camera_pos = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0,N_cam):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vox = prod.clone()
        ax.scatter(vox[i,0,ind], vox[i,1,ind], vox[i,2,ind], s = 1)
        coords_cam = extrinsics[:,:,3]
        ax.scatter(coords_cam[:,0],coords_cam[:,1], coords_cam[:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #ax.set_aspect('equal')
        plt.savefig('test_cloud/camera_pos%d.jpg'%i)
        camera_pos.append(imageio.imread('test_cloud/camera_pos%d.jpg'%i))
        plt.close('all')
    imageio.mimsave('test_cloud/camera_pos.gif', camera_pos)   

def local_proj(xy_coords, N_cam, ind):
    images = []
    for i in range(N_cam):
        fig, ax = plt.subplots(1,1)

        image = xy_coords[i,:,ind]
        x = image[0,:].numpy()
        y = -image[1,:].numpy()
        
        ax.set_aspect('equal')
        plt.scatter(x, y, s = 1)
        plt.savefig('test_cloud/buffer%d.jpg'%i)
        images.append(imageio.imread('test_cloud/buffer%d.jpg'%i))
        plt.close('all')
    imageio.mimsave('test_cloud/virtual_scan.gif', images)