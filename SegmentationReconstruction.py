#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:17:15 2019

@author: alienor
"""

import torch
import torch.nn as nn
from torchvision import models
from collections import defaultdict
import torch.nn.functional as F
from utils.loss import dice_loss
from utils.dataloader import Dataset_3D, init_3D_set
import utils.alienlab as alien
from utils.ply import write_ply, read_ply
import numpy as np
from utils.models import ResNetUNet, my_model_simple, evaluate
import segmentation_models_pytorch as smp


from tqdm import tqdm

import utils.cam_vox as cam_vox
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import utils.vox_to_coord as vtc
import matplotlib.pyplot as plt 
import json
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
#import segmentation_models_pytorch as smp
from PIL import Image
import os
from romidata import io
from romidata.fsdb import _fileset_path
import tempfile


def read_torch(dbfile, ext="pt"):
    """Reads torch tensor from a DB file.
    Parameters
    __________
    dbfile : db.File

    Returns
    _______
    Torch.Tensor
    """
    b = dbfile.read_raw()
    with tempfile.TemporaryDirectory() as d:
        fname = os.path.join(d, "temp.%s"%ext)
        with open(fname, "wb") as fh:
            fh.write(b)
        return torch.load(fname)

def write_torch(dbfile, data, ext="pt"):
    """Writes point cloud to a DB file.
    Parameters
    __________
    dbfile : db.File
    data : TorchTensor
    ext : str
        file extension (defaults to "pt").
    """
    with tempfile.TemporaryDirectory() as d:
        fname = os.path.join(d, "temp.%s"%ext)
        torch.save(data, fname)

        dbfile.import_file(fname)
        

def write_ply_labels(dbfile, data, ext="ply"):
    """Writes point cloud to a DB file.
    Parameters
    __________
    dbfile : db.File
    data : Numpy array
    ext : str
        file extension (defaults to "ply").
    """
    with tempfile.TemporaryDirectory() as d:
        fname = os.path.join(d, "temp.%s"%ext)
        write_ply(fname, data,
          ['x', 'y', 'z', 'labels']) 
        dbfile.import_file(fname)



    
class Dataset(Dataset):

    def __init__(self, image_paths, transform):   # initial logic happens like transform

        self.image_paths = image_paths
        self.transforms = transform

    def __getitem__(self, index):
#io.readimage #get_files
        image = Image.fromarray(io.read_image(self.image_paths[index]))
        
        t_image = self.transforms(image)
        
        t_image = t_image[0:3, :, :]
        return t_image

    def __len__(self):  # return count of sample we have
        return len(self.image_paths)


def voxel_to_pred_by_project(the_shape, torch_voxels, intrinsics, extrinsics, preds_flat, pred_pad, Sx, Sy, xinit, yinit):
        xy_coords = vtc.project_coordinates(torch_voxels, intrinsics, extrinsics, give_prod = False)
        #permute x and y coordinates
        xy_coords[:, 2, :] = xy_coords[:,0,:]
        xy_coords[:, 0, :] = xy_coords[:,1,:]
        xy_coords[:, 1, :] = xy_coords[:,2,:]
        
        coords = vtc.correct_coords_outside(xy_coords, Sx, Sy, xinit, yinit, -1) #correct the coordinates that project outside
        xy_full_flat = vtc.flatten_coordinates(coords, the_shape)
        assign_preds = preds_flat[xy_full_flat].reshape(pred_pad.shape[0], 
                                                xy_full_flat.shape[0]//pred_pad.shape[0], preds_flat.shape[-1])
        del xy_full_flat
        
        assign_preds = torch.sum(assign_preds, dim = 0)
        torch_voxels[:,3] = torch.argmax(assign_preds, dim = 1)
        return torch_voxels
    
def segmentation_space_carving(extrinsics, intrinsics, min_vox, max_vox, num_vox, N_cam, Sx, 
                               Sy, xinit, yinit, cloud_scale, label_names, images_fileset, scan, model_segmentation_name):
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, ' used for images segmentation')


    begin = time.time()
    
    trans = transforms.Compose([
            transforms.CenterCrop((Sx, Sy)),
            transforms.ToTensor()])
    
    image_set = Dataset(images_fileset, transform = trans) 
    batch_size = 1
    sample = 1

    
    loader = DataLoader(image_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    #name0 = 'unet_448_long_train'
    #name0  = 'unet_resnet101_arabidopsis_rotation_table1.2'
    #model_segmentation_name = 'table1_test1_arabidopsis_resnet18_epoch15_rotate_False'
    
    segmentation_folder = '/home/alienor/Documents/Segmentation/'
    model_segmentation = torch.load( segmentation_folder + model_segmentation_name + '.pt')[0]
    try: 
        model_segmentation = model_segmentation.module
    except:
        model_segmentation = model_segmentation

    
    with torch.no_grad():
    
        g = alien.showclass()
        pred_tot = []
        count = 0
        segmentation_folder = scan.get_fileset('Segmented_images', create = True)
        print('Image segmentation by the CNN')
        for inputs in tqdm(loader):
            inputs = inputs.to(device)
            outputs = evaluate(inputs, model_segmentation)
            im_list = [inputs[0][0].cpu().numpy()]
            tl = ['image']
            for i in range(6):
                im_list.append(outputs[0][i].cpu().numpy())
                
            im_list = np.concatenate(im_list, axis = 1)
            im_list = (im_list * 255).astype(np.uint8)
            
            f = segmentation_folder.create_file('segmentation_' + str(count))              
            io.write_image(f, im_list)
            f.set_metadata("image_id", 'segmentation_' + str(count))

            pred_tot.append(outputs)
            count += 1
    
        pred_tot = torch.cat(pred_tot, dim = 0) #All predictions into one tensor
        pred_tot_class = pred_tot[:,1:,:,:]
        #pred_tot_class = nn.MaxPool2d(sf*2)(pred_tot_class)
        #pred_tot_class = nn.Upsample(scale_factor = sf*2)(pred_tot_class)
        pred_tot[:,1:,:,:] = pred_tot_class
    
        pred_pad = torch.zeros((N_cam, len(label_names), xinit, yinit))
        Sbix = pred_tot.shape[2]
        Sbiy = pred_tot.shape[3]
        pred_pad[:,:,(xinit-Sbix)//2:(xinit+Sbix)//2,(yinit-Sbiy)//2:(yinit+Sbiy)//2] = pred_tot #To fit the camera parameters
        #del pred_tot
        pred_pad = pred_pad.permute(0,2,3,1)
    
    
        preds_flat = vtc.adjust_predictions(pred_pad)
        
        basis_voxels = vtc.basis_vox_pipeline(min_vox, max_vox, num_vox[0], num_vox[1], num_vox[2])#List of coordinates  
        vol = torch.from_numpy(basis_voxels)    
        the_shape = torch.Size([N_cam, xinit, yinit, len(label_names)])
        assign_preds = voxel_to_pred_by_project(the_shape, vol, intrinsics, extrinsics, preds_flat, pred_pad,Sx, Sy, xinit, yinit)
        restrict = (assign_preds[:,3] != 6)*(assign_preds[:,3] != 0)
        new_volume = vol[restrict]
        #print(new_volume)
        v = cloud_scale
        full_plant = []
        full_plant.append(new_volume)
        #print(new_volume.shape)
        all_volumes = [new_volume]

        full_plant = [new_volume]
        for i in tqdm(range(2)):
            v = v/2
            shift = torch.tensor([[v,0,0,0],
                                 [0,v,0,0],
                                 [-v,0,0,0],
                                 [0,0,v,0],
                                 [v,0,0,0],
                                 [0,-v,0,0],
                                 [-v,0,0,0],
                                 [0,-v,0,0],
                                 [v, 0, 0,0],
                                 [0,0,-v,0],
                                 [-v, 0,0,0],
                                 [0,0,-v,0],
                                 [v, 0,0,0],
                                 [0,v,0,0],
                                 [0, v,0,0],
                                 [-v, 0,0,0],
                                 [0,-v,0,0],
                                 [-v, 0,0,0],
                                 [0,v,0,0],
                                 [0,0,v,0],
                                 [0,-v,0,0],
                                 [0,v,v,0],
                                 [0,-v,0,0],
                                 [0,-v,0,0],
                                 [0,0,-v,0],
                                 [0,0,-v,0]  
    
             ]).double()
            #print(v)
            for j in range(shift.shape[0]):
                volume_shift = new_volume + shift[j]
                assign_preds = voxel_to_pred_by_project(the_shape, volume_shift, intrinsics, 
                                                    extrinsics, preds_flat, pred_pad,Sx, Sy, xinit, yinit)
                #print(new_volume.shape, assign_preds.shape)
                full_plant.append(assign_preds)
            new_volume = torch.cat(full_plant, dim = 0)
            full_plant = []
        total_points = new_volume

        inds = (total_points[:,3] != 6)*(total_points[:,3] != 0)
        final_volume = total_points[inds].detach().cpu().numpy()
        #final_volume[final_volume[:,3]==5]=4
        #print(final_volume.shape)

        del vol
     
    end = time.time()
    
    save_name = os.path.split(model_segmentation_name)[-1][:-3]
    segmentation_folder = scan.get_fileset('SegmentedPC' , create = True)
    f = segmentation_folder.create_file( save_name + '_SegmentedPC')
    write_ply_labels(f, final_volume)   
    for i, tick in enumerate(label_names):
        f = segmentation_folder.create_file(save_name + '_' + tick)
        write_ply_labels(f, total_points[total_points[:,3] == i].detach().cpu().numpy())   


    
    print('3D reconstruction successfully achieved in ', end - begin, 's')
    
    return final_volume
        

    
    
    '''
    DEBUG: projecct the volume to carve on the images
    
    #Camera projection
    basis_voxels = vtc.basis_vox(min_vox, num_vox[0],
                                 num_vox[1], num_vox[2])*cloud_scale#List of coordinates  

    torch_voxels = torch.from_numpy(basis_voxels)
    #Perspective projection
    xy_coords = vtc.project_coordinates(torch_voxels, intrinsics, extrinsics, give_prod = False)
    
    #permute x and y coordinates
    xy_coords[:, 2, :] = xy_coords[:,0,:]
    xy_coords[:, 0, :] = xy_coords[:,1,:]
    xy_coords[:, 1, :] = xy_coords[:,2,:]
    for k in range(72):
        xy_coords[k,0, xy_coords[k,0] > xinit] = 0
        xy_coords[k,0, xy_coords[k,0] < 0] = 0
        
        xy_coords[k,1, xy_coords[k,1] > yinit] = 0
        xy_coords[k,1, xy_coords[k,1] < 0] = 0
               
        pred_pad = torch.zeros((N_cam, len(label_names), xinit, yinit))
        pred_tot_cop = pred_tot.clone()
        pred_pad[:,:,(xinit-Sx)//2:(xinit+Sx)//2,(yinit-Sy)//2:(yinit+Sy)//2] = pred_tot_cop #To fit the camera parameters
        pred_pad[k,:,xy_coords[k,0,:].long(), xy_coords[k,1,:].long()] = 1
        g = alien.showclass()
        g.save_folder = "Segmented_images/"
        g.save_name = '%d'%k
        g.title_list = ['padded prediction (class: background)']
        g.showing(pred_pad[k,0,:,:])

     '''   



    