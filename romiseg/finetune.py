#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:50:33 2019

@author: alienor
"""

#import open3d
import argparse
import appdirs
import glob
import numpy as np
import os
from PIL import Image
import torch
import subprocess 
import tkinter as tk
from tkinter.filedialog import askopenfilenames
root = tk.Tk()
root.withdraw()
from tkinter import filedialog


from romiseg.train_cnn import cnn_train
from romiseg.utils.train_from_dataset import save_and_load_model
from romiseg.utils.active_contour import run_refine_romidata
from romidata import fsdb, io

import toml

def create_folder_if(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        open('romidb', 'w').close()
        
default_config_dir = '/home/alienor/Documents/scanner-meta-repository/Scan3D/config/segmentation2d_arabidopsis.toml'

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', default=default_config_dir,
                    help='config dir, default: %s'%default_config_dir)


args = parser.parse_args()

print(args.config)

param_pipe = toml.load(str(args.config))

direc = param_pipe['Finetune']
try:
    directory_weights = direc['directory_weights']
except:
    directory_weights = appdirs.user_cache_dir()
try:    
    tsboard = direc['tsboard'] + '/finetune'
except:
    tsboard = appdirs.user_cache_dir()

param2 = param_pipe['Segmentation2D']
labels = param2['labels']
Sx = param2['Sx']
Sy = param2['Sy']
learning_rate = param2['learning_rate']
model_segmentation_name = param2['model_segmentation_name']



finetune = param_pipe['Finetune']
directory_dataset = finetune['directory_images']
finetune_epochs = finetune['finetune_epochs']
batch_size = finetune['batch']

subprocess.call(["gio", "mount", "ssh://db.romi-project.eu"])

#if directory_dataset == 'complete here':
    #directory_dataset = filedialog.askdirectory(initialdir="/home/", title='create folder to save fine-tuning images')
    #create_folder_if(directory_dataset)
    #param2['directory_images'] = directory_dataset
    
#Save folder
#directory_weights = appdirs.user_cache_dir()

#if directory_weights == 'complete here':
#    directory_weights = filedialog.askdirectory(initialdir="/home/", title='create folder to save fine-tuning weights')
#    create_folder_if(directory_weights)
#    param['directory_weights'] = directory_weights
#directory_images = '/home/alienor/Documents/database/FINETUNE'
#directory_weights = '/home/alienor/Documents/database/WEIGHTS'



files = askopenfilenames(initialdir = os.path.split("/home/")[0], 
                         title = 'Select some pictures to annotate')
lst = list(files)

if len(lst) > 0:
    host_scan = files[0].split('/')[-3] 

    db = fsdb.FSDB(directory_dataset + '/train/')
    db.connect()
    
    scan = db.get_scan(host_scan, create=True)
    fileset = scan.get_fileset('images', create = True)
    
    imgs = np.sort(files)
    
    
    for i, path in enumerate(imgs):
        im_name = host_scan + '_' + os.path.split(path)[1][:-4]
        

        im = np.array(Image.open(path))
        f_im = fileset.create_file(im_name + '_rgb')
        f_im.set_metadata('shot_id', im_name)
        f_im.set_metadata('channel', 'rgb')
        io.write_image(f_im, im)
        
        im_save = fsdb._file_path(f_im)
        subprocess.run(['labelme', im_save, '-O', im_save, '--labels', labels])
        
        npz = run_refine_romidata(im_save, 1, 1, 1, 1, 1, class_names = labels.split(',')[1:], 
                           plotit = im_save)
        print(npz)
        
        f_label = fileset.create_file(im_name + '_segmentation')
        f_label.set_metadata('shot_id', im_name)
        f_label.set_metadata('channel', 'segmentation')
        io.write_npz(f_label, npz)
    db.disconnect()
 
    
subprocess.run(["rsync", "-av", directory_dataset, appdirs.user_cache_dir()])
directory_dataset = appdirs.user_cache_dir()
label_names = labels.split(',')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = save_and_load_model(directory_weights, model_segmentation_name).to(device)

# freeze backbone layers
for l in model.base_layers:
    for param in l.parameters():
        param.requires_grad = False
       
model = cnn_train(directory_weights, directory_dataset, label_names, tsboard, batch_size, finetune_epochs,
                    model, Sx, Sy, showit = True)

model_name =  model_segmentation_name[:-3] + os.path.split(directory_dataset)[1] +'_%d_%d_'%(Sx,Sy)+ 'finetune_epoch%d.pt'%finetune_epochs

torch.save(model, directory_weights + '/' + model_name)


param2['model_segmentation_name'] = model_name

text = toml.dumps(param_pipe)
   

text_file = open(args.config, "w")
text_file.write(text)
text_file.close()

print('/n')
print("You have fine-tunned the segmentation network with the images you manually annotated.")
print("The pipeline should work better on your images now, let's launch it again")

subprocess.call(["gio", "mount", "-u", "ssh://db.romi-project.eu"])
