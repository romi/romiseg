#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:42:25 2019

@author: alienor
"""

import bpy
import tempfile

import numpy as np
import sys
import glob
import os

import sys
argv = sys.argv
argv = argv[argv.index("--") + 1:]

objs = bpy.data.objects
objs.remove(objs["Cube"], True)

materials = ['Color_0', 'Color_1', 'Color_2', 'Color_7', 'Color_8']
colors = [(1, 0.2, 1), (0.2, 0.4, 0.7), (0.4, 0.6, 0.9) , (0.6, 0.8, 0.5) , (0.8, 1., 0.3)]
n = int(argv[0])
directory = "../blender_virtual_scanner/data/"
filename = "arabidopsis_%d.obj"%n

imported_object = bpy.ops.import_scene.obj(filepath=directory + filename)

for i, key in enumerate(materials):
    if key in bpy.data.materials.keys():
        mat = bpy.data.materials[key]
        mat.diffuse_color = colors[i]
    

target_dir = 'LEARNING/ground_truth_3D/'
target_file = os.path.join(target_dir, 'arabidopsis_3D_GT_%03d.obj'%n)

bpy.ops.export_scene.obj(filepath=target_file)
print(obj_object.material_slots)