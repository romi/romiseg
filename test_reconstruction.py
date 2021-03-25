"""
Created on Thu Nov 21 09:18:24 2019

@author: alienor
"""
import open3d

import os
import argparse
import toml
from PIL import Image
import numpy as np

import segmentation_models_pytorch as smp
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import torch.optim as optim
#from torchvision import models

from plantdb import io
from plantdb import fsdb

from romiseg.utils.train_from_dataset import train_model
from romiseg.utils.dataloader_finetune import plot_dataset
from romiseg.utils import segmentation_model

import romiseg.utils.vox_to_coord as vtc
from romiseg.utils.generate_volume import generate_ground_truth
from romiseg.utils.ply import read_ply, write_ply



pcd_loc = '/home/alienor/Documents/blender_virtual_scanner/data/COSEG/guitar/'

default_config_dir = "/home/alienor/Documents/scanner-meta-repository/Segmentation/romiseg/parameters_train.toml"


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

generate_ground_truth(directory_dataset + '/val/', pcd_loc, coord_file_loc, 
                           Sx, Sy, N_vox, label_names)

db = fsdb.FSDB(directory_dataset)
db.connect()
scan = db.get_scans()[0]
print('Reference scan used to generate data: ', scan.id)

masks = scan.get_fileset('images')

gt = masks.get_files(query={'channel':'segmentation'})
pred_tot = []
for i, seg in enumerate(gt):
    seg = io.read_npz(seg)
    class_pred = []
    for j, class_name in enumerate(seg.files):
        class_pred.append(seg[seg.files[j]])
    class_pred = np.stack(class_pred, axis = 0)
    pred_tot.append(class_pred)

pred_tot = torch.Tensor(pred_tot)

pred_tot = pred_tot.permute(0,2,3,1)//255
preds_flat = vtc.adjust_predictions(pred_tot)


xy_full_flat = torch.load(coord_file_loc + '/coords.pt')
voxels = torch.load(coord_file_loc + '/voxels.pt')

assign_preds = preds_flat[xy_full_flat].reshape(pred_tot.shape[0], 
                                        xy_full_flat.shape[0]//pred_tot.shape[0], preds_flat.shape[-1])
assign_preds = assign_preds[:,:,0:-1]
assign_preds = torch.log(assign_preds)
assign_preds = torch.sum(assign_preds, dim = 0)

#assign_preds = torch.sum(assign_preds, dim = -1)
preds_max = torch.max(assign_preds, dim = -1).values
voxels[:,3] = torch.argmax(assign_preds, dim = -1)
voxels = voxels[preds_max >= 0]
#voxels = voxels[voxels[:,3] != 0]

"""
    
assign_preds = preds_flat[xy_full_flat].reshape(pred_tot.shape[0], 
                                        xy_full_flat.shape[0]//pred_tot.shape[0], preds_flat.shape[-1])
assign_preds = torch.sum(torch.log(assign_preds + 0.01), dim = 0)
#assign_preds[:,0] *= 2
voxels[:,3] = torch.argmax(assign_preds[:,:-1], dim = 1)
voxels = voxels[voxels[:,3] != 0]

"""
write_ply(coord_file_loc +  '/test_rec.ply', [voxels.numpy()],
      ['x', 'y', 'z', 'label'])
    
    


db.disconnect()