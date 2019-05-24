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

if __name__ == "__main__":
    W = 1920
    H = 1080

    for i in range(0, 50):

        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'train')
       
    for i in range(51, 80):
        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'val')
    
    for i in range(81, 82):
        tool = virtual_scan(w = W, h = H)
        tool.image_and_label(i, N = 72, R= 35, z = 60, mode = 'test')
