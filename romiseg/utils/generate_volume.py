#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:50:24 2019

@author: alienor
"""
from romidata import io, fsdb
import torch
import romiseg.utils.vox_to_coord as vtc
import numpy as np


import argparse
import toml

from romidata import io
from romidata import fsdb

from romiseg.utils.train_from_dataset import train_model
from romiseg.utils.dataloader_finetune import plot_dataset
from romiseg.utils import segmentation_model

import romiseg.utils.vox_to_coord as vtc
import romiseg.utils.generate_3D_ground_truth as gt_vox


pcd_loc = '/home/alienor/Documents/blender_virtual_scanner/data/COSEG/guitar/'

default_config_dir = "../parameters_train.toml"

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--config', dest='config', default=default_config_dir,
                    help='config dir, default: %s'%default_config_dir)


args = parser.parse_args()



param_pipe = toml.load(args.config)

direc = param_pipe['Directory']

path = direc['path']
directory_weights = path + direc['directory_weights']
model_segmentation_name = direc['model_segmentation_name']
tsboard = path +  direc['tsboard']
directory_dataset = path + direc['directory_dataset']


param2 = param_pipe['Segmentation2D']

label_names = param2['labels'].split(',')


Sx = param2['Sx']
Sy = param2['Sy']

epochs = param2['epochs']
batch_size = param2['batch']

learning_rate = param2['learning_rate']


param3 = param_pipe['Reconstruction3D']
N_vox = param3['N_vox']
coord_file_loc = path + param3['coord_file_loc']


def build_bounding_box(scan, N_vox):
    bbox = scan.metadata['bounding_box']
    disp = scan.metadata['displacement']
    bkeys = list(bbox.keys())
    dkeys = list(disp.keys())
    min_vox = np.array([bbox[bkeys[i]][0] - disp[dkeys[i]] for i in range(len(bkeys))])
    max_vox = np.array([bbox[bkeys[i]][1] - disp[dkeys[i]] for i in range(len(bkeys))])
    bound = max_vox - min_vox
    xyz = bound[0] * bound[1] * bound[2]        
    
    cloud_scale = (xyz/N_vox)**(1/3)
    num_vox = (max_vox - min_vox)//(cloud_scale) + 1
    num_vox = num_vox.astype(int)
    return min_vox - 10, max_vox + 10, num_vox, cloud_scale

    
def read_intrinsics(camera_model):
    """
    Read camera intrinsics
    input: scan id fsdb
    output: -xinit, yinit [int] image height and width
            -intrinsics [torch tensor] 3x3 matrix of camera intrinsics: focal and optical center position on images
    """

    #image dimensions         
    xinit = camera_model['height'] #image width
    yinit = camera_model['width'] #image height

    focal = camera_model['params'][0:4] #camera intrinsics

    #buiding the extrinsics matrix
    intrinsics = torch.zeros((1, 3, 3))
    intrinsics[:,0,0] = focal[0]        
    intrinsics[:,1,1] = focal[1]
    intrinsics[:,0,2] = focal[2]
    intrinsics[:,1,2] = focal[3]
    intrinsics[:,2,2] = 1

    return xinit, yinit, intrinsics

def read_extrinsics(images_fileset, N_cam):
    """
    Read camera extrinsics
    input: scan id fsdb
    output: -extrinsics [torch tensor] 3x4 matrix of camera extrinsics:
        rotation and translation of the object with respect to the camera
    """

    #Collect camera positions, there are as many as images in the scan folder. Located in "images.json" file

    #camera extrinsics
    extrinsics = torch.zeros((N_cam, 3, 4))

    #Associate camera position to corresponding image
    for i, fi in enumerate(images_fileset):
        pose = fi.metadata['camera']    
        rot = pose['rotmat']
        tvec = pose['tvec']
        extrinsics[i][:3,:3] = torch.Tensor(rot)
        extrinsics[i][:,3] = torch.Tensor(tvec)  


    return extrinsics


def build_voxel_volume(scan, coord_file_loc, extrinsics, intrinsics, min_vox, max_vox, num_vox, N_cam  = 72, cloud_scale = 2,
                       Sx= 896, Sy = 896, xinit = 896, yinit = 896, label_num = 6):
    #Voxel representation of the point cloud
    basis_voxels = vtc.basis_vox_pipeline(min_vox, max_vox, num_vox[0], num_vox[1], num_vox[2])#List of coordinates  
    
    print('generation of the 3D volume to carve')

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

    volume = scan.get_fileset('volume', create=True)
    coord_file = volume.get_file('coords', create=True)
    io.write_torch(coord_file, xy_full_flat)
    voxel_file = volume.get_file('voxels', create=True)
    io.write_torch(voxel_file, torch_voxels)
    torch.save(xy_full_flat, coord_file_loc + 'coords.pt')
    torch.save(torch_voxels, coord_file_loc + 'voxels.pt')
    del xy_full_flat
    del coords
    
    return torch_voxels

def generate_volume(directory_dataset, coord_file_loc, Sx, Sy, N_vox, label_names):
    
    db = fsdb.FSDB(directory_dataset)
    db.connect()
    scan = db.get_scans()[0]
    
    images = scan.get_fileset('images').get_files(query = {'channel' : 'rgb'})
    camera = images[0].metadata['camera']['camera_model']
    xinit, yinit, intrinsics = read_intrinsics(camera)
    N_cam = len(images)
    extrinsics = read_extrinsics(images, N_cam)
    min_vox, max_vox, num_vox, cloud_scale = build_bounding_box(scan, N_vox)
    
    torch_voxels = build_voxel_volume(scan, coord_file_loc, extrinsics, intrinsics, min_vox, max_vox, num_vox, N_cam, cloud_scale,
                           Sx, Sy, xinit, yinit, len(label_names))
    
        
    db.disconnect()
    return torch_voxels, num_vox, min_vox, cloud_scale
    
torch_voxels, num_vox, min_vox, cloud_scale = generate_volume(directory_dataset, coord_file_loc, Sx, Sy, N_vox, label_names)
    
#def generate_ground_truth(directory_dataset, pcd_loc, torch_voxels ):
db = fsdb.FSDB(directory_dataset)
db.connect()
for scan in db.get_scans():
    fetch_pcd = np.load(pcd_loc + scan.id + ".npz")
    pcd = fetch_pcd[fetch_pcd.files[0]]   
    vox = gt_vox.gt_pc_to_vox(pcd, torch_voxels, num_vox, min_vox, cloud_scale )
    gt_3D = scan.get_fileset('ground_truth_3D', create = True)
    f = gt_3D.get_file('voxel_classes', create = True)
    io.write_torch(f, vox)
db.disconnect()

