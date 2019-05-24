#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:39:10 2019

@author: alienor
"""

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

from read_model import *

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
import imageio
#%%
if __name__ == "__main__":

    
    #FROM A POINT CLOUD
    
    n = 4
    filepath = r"point_clouds/arabidopsis_%d"%n
    density = 10 #pts/cm³
    cloud_scale = 1
    
    
    N = int(65./cloud_scale)
    #Voxel representation of the point cloud
    min_vec = [int(-40/cloud_scale), int(-40/cloud_scale),int(-5/cloud_scale)] #Limit of the cloud
    basis_voxels = basis_vox(min_vec, N, N, 3*N) #List of coordinates  

    #Load mesh as point cloud by calling cloudcompare
    points = load_mesh_as_pcd(filepath, n, density, cloud_scale)
#%% 
    voxels = fill_vox(points, basis_voxels, cloud_scale, min_vec, N, N, 3*N) #Fill the coordinates
    #test
    #plot_3D(voxels) #Check the 3D point cloud representation in the voxels
 
#%%
    if True: 
        import json
        with open('images.json', 'r') as f:
            pose = json.load(f)
        
        N_cam = 72
        extrinsics = torch.zeros((N_cam, 3, 4))
        for i in range(N_cam):
            rot = pose[str(i+1)]['rotmat']
            extrinsics[i][:3,:3] = torch.Tensor(rot)
            trans = pose[str(i+1)]['tvec']
            extrinsics[i][:,3] = torch.Tensor(trans)/10.
                
        with open('cameras.json', 'r') as f:
            focal = json.load(f)
        focal = focal['1']['params']
        
        intrinsics = torch.zeros((1, 3, 3))
        intrinsics[:,0,0] = focal[0]
        intrinsics[:,1,1] = focal[0]
        intrinsics[:,0,2] = focal[1]
        intrinsics[:,1,2] = focal[2]
        intrinsics[:,2,2] = 1
        #intrinsics = intrinsics.permute(0,2,1)
    
    if False:
        path = '../../test_colmap/dense/0/sparse'
        ext = '.bin'
        cameras, images, points3D = read_model(path, ext)
        
        N_cam = 72
        extrinsics = torch.zeros((N_cam, 3, 4))
        for i in range(N_cam):
            extr = images[i+1]
            rot = extr.qvec2rotmat()
            extrinsics[i][:3,:3] = torch.Tensor(rot.T)
            trans = -np.dot(rot.T, images[i+1].tvec)
            extrinsics[i][:,3] = torch.Tensor(trans)
                
        focal = cameras[1].params
        
        intrinsics = torch.zeros((1, 3, 3))
        intrinsics[:,0,0] = focal[0]
        intrinsics[:,1,1] = focal[1]
        intrinsics[:,0,2] = focal[2]
        intrinsics[:,1,2] = focal[3]
        intrinsics[:,2,2] = 1
    
#%%   
    #Camera projection
 
    torch_voxels = torch.from_numpy(voxels)

    #Perspective projection
    prod, xy_coords = project_coordinates(torch_voxels, intrinsics, extrinsics, give_prod = True)

    xy_coords[:, 2, :] = xy_coords[:,0,:] #Exchange x and y coordinates
    xy_coords[:, 0, :] = xy_coords[:,1,:]
    xy_coords[:, 1, :] = xy_coords[:,2,:]

    ind = torch_voxels[:,3]==1
    
    #test
    #move_camera(torch_voxels, extrinsics, N_cam, prod)  
    #local_proj(xy_coords, N_cam, ind)
    Sx = 1080.
    Sy = 1920. 
    s = 1./( cloud_scale)   
    maxi = torch.tensor([s * Sx * cloud_scale , s * Sy * cloud_scale])
  
    
    #View the images 
#%%
    if True:
        i = 1
        images = views_to_images(torch_voxels, xy_coords, s, cloud_scale, N_cam, maxi)
        imbis = images.copy()
        for i in range(N_cam//2):
            mask = images[i][1:,1:].numpy()
            #im = np.array(Image.open('segmentation_arabidopsis/arabidopsis004/labels/arabidopsis004_image%03d.png'%i))
            #im = im
            #im[mask == 1] = 255
            #imbis[i] = im
       
        imageio.mimsave('cloud_sample/virtual_scan_torch%d_ims.gif'%n, images)
        #plt.figure()#
        #plt.imshow(im)
        #plt.show()
        
#%%
        vol = torch.from_numpy(basis_vox(min_vec, N, N, 3*N))
        coords = xy_coords.clone().int()
        for i in range(N_cam):
            coords_i = coords[i]
            ind_0 = coords_i[0,:]>0
            #print(np.count_nonzero(ind_0))
            #coords_i = coords[i][:, ind_0]
            ind_1 = coords_i[1,:]>0
            #print(np.count_nonzero(ind_1))
            #coords_i = coords_i[:, ind_1]
            ind_2 = coords_i[0,:] < int(maxi[0])
            #print(torch.max(coords_i[0,:]))
            #coords_i = coords_i[:, ind_2]
            ind_3 = coords_i[1,:] < int(maxi[1])
            #print(torch.max(coords_i[1,:]))
            #coords_i = coords_i[:, ind_3]            
            ind_tot = ind_0 * ind_1 * ind_2 * ind_3
            coords_i = coords_i[:,ind_tot]
            im = images[i].type(torch.DoubleTensor)
            coords_i = coords_i.type(torch.LongTensor)
            vol[ind_tot,3] += im[coords_i[0], coords_i[1]]
            
        points = vol[vol[:,3] > 2, :].numpy()
        write_ply('rebuilt_cloud.ply', [points/10],
          ['x', 'y', 'z', 'label'])
        
#%%
        
    if False:
        vol = torch.from_numpy(basis_vox(min_vec, N, N, 3*N))
        coords = xy_coords.clone().int()
        maxi = torch.tensor([Sx , Sy])
        full_proj = []
        for i in range(N_cam):
            coords_i = coords[i]
            ind_0 = coords_i[0,:]>0
            #print(np.count_nonzero(ind_0))
            #coords_i = coords[i][:, ind_0]
            ind_1 = coords_i[1,:]>0
            #print(np.count_nonzero(ind_1))
            #coords_i = coords_i[:, ind_1]
            ind_2 = coords_i[0,:] < int(maxi[0])
            #print(torch.max(coords_i[0,:]))
            #coords_i = coords_i[:, ind_2]
            ind_3 = coords_i[1,:] < int(maxi[1])
            #print(torch.max(coords_i[1,:]))
            #coords_i = coords_i[:, ind_3]            
            ind_tot = ind_0 * ind_1 * ind_2 * ind_3
            coords_i = coords_i[:,ind_tot].long()
            #im = pred_pad[i,1].type(torch.DoubleTensor)
            #coords_i = coords_i.type(torch.LongTensor)
            #vol[ind_tot,3] += im[coords_i[0], coords_i[1]]
            
            #plt.figure()
            im = torch.zeros((1080,1920))
            im[coords_i[1], coords_i[0]] = 1
            full_proj.append(im.numpy())
        
        imageio.mimsave('cloud_sample/proj.gif', full_proj)
 
       # points = vol[vol[:,3] > 2, :].numpy()
       # write_ply('rebuilt_cloud.ply', [points/10],
        #  ['x', 'y', 'z', 'label'])