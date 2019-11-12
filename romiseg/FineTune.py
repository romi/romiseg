#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:50:33 2019

@author: alienor
"""

#import open3d

import numpy as np
import os
from PIL import Image

import subprocess 
import tkinter as tk
from tkinter.filedialog import askopenfilenames
root = tk.Tk()
root.withdraw()
from tkinter import filedialog


from romiseg.utils.train_from_dataset import fine_tune_train
from romiseg.utils.active_contour import run_refine
from romiseg.utils.alienlab import create_folder_if

import toml



#with open("finetune_intro.md", "r") as fh:
#    long_description = fh.read()
def finetune():
        pipeline = '/home/alienor/Documents/Scan3D/script/pipeline.toml'
        param_pipe = toml.load(pipeline)
        param = param_pipe['Segmentation2D']
        directory_images = param['directory_images']
        directory_weights = param['directory_weights']
        model_segmentation_name = param['model_segmentation_name']
        Sx = param['Sx']
        Sy = param['Sy']
        finetune_epochs = param['finetune_epochs']
        
        if directory_images == 'complete here':
            directory_images = filedialog.askdirectory(initialdir="/home/", title='create folder to save fine-tuning images')
            create_folder_if(directory_images)
            param['directory_images'] = directory_images
            
        if directory_weights == 'complete here':
            directory_weights = filedialog.askdirectory(initialdir="/home/", title='create folder to save fine-tuning weights')
            create_folder_if(directory_weights)
            param['directory_weights'] = directory_weights
        #directory_images = '/home/alienor/Documents/database/FINETUNE'
        #directory_weights = '/home/alienor/Documents/database/WEIGHTS'
        
        create_folder_if(directory_images + '/images')
        create_folder_if(directory_images + '/labels')
        
        scan = 'folder'
        
        
        files = askopenfilenames(initialdir = os.path.split(directory_images)[0], 
                                 title = 'Select some pictures to annotate')
        lst = list(files)

        if len(lst) > 0:
            imgs = np.sort(files)
            scan = os.path.split(os.path.split(os.path.split(imgs[0])[0])[0])[1]
            
            labels = 'stem,peduncle,flower,background,fruit,leaf'
            
            for i, path in enumerate(imgs):
                im_name = scan + '_' + os.path.split(path)[1][:-4]
                im_name =  os.path.split(path)[1][:-4]
                save_im = directory_images + '/images/' + im_name + '.jpg'
                save_labels = directory_images + '/labels/' + im_name + '.png'
                im = Image.open(path)
                im.save(save_im, 'JPEG')
                subprocess.run(['labelme', save_im, '-O', save_im, '--labels', labels])
                
                run_refine(save_im, 1, 1, 1, 1, 1, 
                                   plotit = save_labels)
        
         
        labels_names = ['background', 'flowers', 'peduncle', 'stem', 'leaves', 'fruits']
     
        model, new_model_name = fine_tune_train(directory_images, directory_images, directory_weights,
                        labels_names, scan, model_segmentation_name, Sx, Sy, finetune_epochs, scan)
        
        param['model_segmentation_name'] = new_model_name
        
        text = toml.dumps(param_pipe)
           
        
        text_file = open(pipeline, "w")
        text_file.write(text)
        text_file.close()
        
        print('/n')
        print("You have fine-tunned the segmentation network with the images you manually annotated.")
        print("The pipeline should work better on your images now, let's launch it again")
        