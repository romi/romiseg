#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:06:29 2019

@author: alienor
"""


import torch

from utils.ply import *
import utils.proj_point_cloud as ppc
import utils.vox_to_coord as vtc
import matplotlib.pyplot as plt
import utils.tests as test


def to_sparse(x):
    """ converts dense tensor x to sparse format """


    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices
    values = x[[indices[i] for i in range(indices.shape[0])]]
    return indices, values, x.size()


def gt_pc_to_vox(filename):
    cloud_scale = 0.5
    N = int(65/cloud_scale)
    min_vec = [int(-40/cloud_scale), int(-40/cloud_scale),int(-5/cloud_scale)] #Limit of the cloud
    basis_voxels = vtc.basis_vox(min_vec, N, N, N)/2 #List of coordinates  
    
    #filename = "point_clouds/arabidopsis_%d_GT.ply"%n
    
    gt = read_ply(filename)
    gt_points = np.vstack((gt['x'], gt['y'], gt['z'], gt['green'])).T
    
    voxels = ppc.fill_vox(gt_points, basis_voxels, cloud_scale, min_vec, N, N, N) #Fill the coordinates
    
    voxels = torch.tensor(voxels)
    
    
    
    a, b, c = to_sparse(voxels[:,3])
    a = a[:,0].long()
    b = b.long()//50
    vox_sparse = torch.stack([a, b])
    torch.save(vox_sparse, filename[:-4] + '.pt')
    
    #Convert back:
    #vox = torch.sparse.FloatTensor(vox_sparse[0].unsqueeze(0), vox_sparse[1], 
    #size = torch.Size([basis_voxels.shape[0]])).to_dense()