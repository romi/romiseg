#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:08:39 2019

@author: alienor
"""
#path = '/home/alienor/Documents/blender_virtual_scanner'
#import sys
#sys.path.append(path)
from save_images import virtual_scan 
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os 
from utils.ply import *
import utils.generate_3D_ground_truth as gt_vox

import utils.proj_point_cloud as ppc
import utils.vox_to_coord as vtc
import matplotlib.pyplot as plt
import utils.tests as test


if __name__ == "__main__":
    W = 1920
    H = 1080


#BLENDER IMAGES
if False: 
    for i in range(0, 50):

        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'train')
       
    for i in range(50, 80):
        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'val')
    
    for i in range(80, 100):
        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'test')


#%%
#CHANGE MATERIALS
NN = 100
if True: 
    for i in range(NN):
        openFile=os.system("blender -b -P utils/change_materials.py -- %d"%i)
        
#%%
#MESH TO POINT CLOUD
if True:
    for n in range(NN):
        directory = "LEARNING/ground_truth_3D/"
        filepath = directory + "arabidopsis_3D_GT_%03d"%n
        filename = filepath + ".obj"
        savename = filepath + ".ply"
        
        density = 100 
        
        openFile=os.system("CloudCompare -SILENT -O " + filename + 
                       " -C_EXPORT_FMT PLY -AUTO_SAVE OFF -SAMPLE_MESH DENSITY %d -SAVE_CLOUDS FILE "%density + savename)

        
        gt_vox.gt_pc_to_vox(savename)

        

    