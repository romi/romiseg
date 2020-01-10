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

import subprocess 
import tkinter as tk
from tkinter.filedialog import askopenfilenames
root = tk.Tk()
root.withdraw()
from tkinter import filedialog



from romiseg.utils.train_from_dataset_romidata import fine_tune_train
from romiseg.utils.active_contour import run_refine
from romiseg.utils.alienlab import create_folder_if
from romidata import fsdb, io

import toml


default_config_dir = '/home/alienor/Documents/Scan3D/config/segmentation2d.toml'

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', default=default_config_dir,
                    help='config dir, default: %s'%default_config_dir)


args = parser.parse_args()

print(args.config)

param_pipe = toml.load(str(args.config))

param = param_pipe['Segmentation2D']
directory_images = param['directory_images']
#directory_weights = param['directory_weights']
model_segmentation_name = param['model_segmentation_name']
Sx = param['Sx']
Sy = param['Sy']
labels = param['label_names']

finetune_epochs = param['finetune_epochs']

if directory_images == 'complete here':
    directory_images = filedialog.askdirectory(initialdir="/home/", title='create folder to save fine-tuning images')
    create_folder_if(directory_images)
    param['directory_images'] = directory_images
    
#Save folder
directory_weights = appdirs.user_cache_dir()

#if directory_weights == 'complete here':
#    directory_weights = filedialog.askdirectory(initialdir="/home/", title='create folder to save fine-tuning weights')
#    create_folder_if(directory_weights)
#    param['directory_weights'] = directory_weights
#directory_images = '/home/alienor/Documents/database/FINETUNE'
#directory_weights = '/home/alienor/Documents/database/WEIGHTS'

create_folder_if(directory_images + '/images')

scan = 'folder'


#files = askopenfilenames(initialdir = os.path.split(directory_images)[0], 
 #                        title = 'Select some pictures to annotate')

imdir = "/home/alienor/Documents/database/FINETUNE/images"
files = glob.glob(imdir + '/*.jpg')

lst = list(files)

if len(lst) > 0:
    host_scan = files[0].split('/')[-3]

    db = fsdb.FSDB(directory_images)
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
        #subprocess.run(['labelme', im_save, '-O', im_save, '--labels', labels])
        
        npz = run_refine_romidata(im_save, 1, 1, 1, 1, 1, class_names = labels.split(',')[1:], 
                           plotit = im_save)
        print(npz)
        
        f_label = fileset.create_file(im_name + '_segmentation')
        f_label.set_metadata('shot_id', im_name)
        f_label.set_metadata('channel', 'segmentation')
        io.write_npz(f_label, npz)
    db.disconnect()
 
labels_names = labels.split(',')

print(directory_images)

model, new_model_name = fine_tune_train(directory_images, directory_images, directory_weights,
                labels_names, scan, model_segmentation_name, Sx, Sy, finetune_epochs, scan)

param['model_segmentation_name'] = new_model_name

text = toml.dumps(param_pipe)
   

text_file = open(args.config, "w")
text_file.write(text)
text_file.close()

print('/n')
print("You have fine-tunned the segmentation network with the images you manually annotated.")
print("The pipeline should work better on your images now, let's launch it again")
