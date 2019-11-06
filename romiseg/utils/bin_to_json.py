#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:53:28 2019

@author: alienor
"""

import colmap

fname = '../data/lego/train/camera/cameras.txt'
with open(fname,'rb') as f:
    fileContent = f.read()     