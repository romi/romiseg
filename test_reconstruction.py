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

from romidata import io
from romidata import fsdb

from romiseg.utils.train_from_dataset import train_model
from romiseg.utils.dataloader_finetune import plot_dataset
from romiseg.utils import segmentation_model

import romiseg.utils.vox_to_coord as vtc
from romiseg.utils.generate_volume import generate_volume


default_config_dir = "romiseg/parameters_train.toml"

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


db = fsdb.FSDB(directory_dataset)
db.connect()
scan = db.get_scans()[0]
print(scan.id)

masks = scan.get_fileset('images')

gt = masks.get_files(query={'channel':'segmentation'})
pred_tot = []
for i, seg in enumerate(gt):
    seg = io.read_npz(seg)
    seg = seg[seg.files[0]]
    pred_tot.append(seg)

pred_tot = np.array(pred_tot)