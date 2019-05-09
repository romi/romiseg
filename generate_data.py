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

    for i in range(1,50):
        tool = virtual_scan(w = 896, h = 448)
        tool.image_and_label(i, N = 10, mode = 'train')
       
    for i in range(51, 80):
        tool = virtual_scan(w = 896, h = 448)
        tool.image_and_label(i, N = 10, mode = 'val')
    
    for i in range(81, 100):
        tool = virtual_scan(w = 896, h = 448)
        tool.image_and_label(i, N = 10, mode = 'test')
