#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:06:29 2019

@author: alienor
"""


import torch

import numpy as np
from romiseg.utils.ply import read_ply, write_ply
import romiseg.utils.proj_point_cloud as ppc
import romiseg.utils.vox_to_coord as vtc
import matplotlib.pyplot as plt


def to_sparse(x):
    """ converts dense tensor x to sparse format """


    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices
    values = x[[indices[i] for i in range(indices.shape[0])]]
    return indices, values, x.size()


def gt_pc_to_vox(filename, torch_voxels, num_vox, min_vec, cloud_scale, savename=None):
    

    [w, h, l] = num_vox
    voxels, v = ppc.fill_vox(filename, torch_voxels, cloud_scale, min_vec, w, h, l) #Fill the coordinates

    voxels = torch.tensor(voxels)
    
    
    
    #a, b, c = to_sparse(voxels[:,3])
    #a = a[:,0].long()
    #b = b.long()//50
    #vox_sparse = torch.stack([a, b])
    #if savename==None:
    #    torch.save(vox_sparse, filename[:-4] + '.pt')
    #else: 
    #    torch.save(vox_sparse, savename + '.pt')

    return voxels
    #Convert back:
    #vox = torch.sparse.FloatTensor(vox_sparse[0].unsqueeze(0), vox_sparse[1], 
    #size = torch.Size([basis_voxels.shape[0]])).to_dense()