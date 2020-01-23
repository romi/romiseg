#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:20:54 2019

@author: alienor
"""

import os
import sys


import re

import platform
import subprocess

import site
from distutils.sysconfig import get_python_inc

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

install_requires=[
        'torchvision',
        'appdirs',
        'Pillow==6.1',
        'tqdm',
        'torch',        
        'requests',
        'mako',
        'tensorboard',
        'future',
        'labelme',       
        
        
    ]


s = setup(
    name='romiseg',
    version='0.0.1',
    scripts=['romiseg/finetune.py', 'romiseg/train_cnn.py'],
    packages=find_packages(),
    author='Alienor Lahlou',
    author_email='alienor.lahlou@espci.org',
    description='Image multiclass segmentation using CNN models trained on virtual images (PyTorch)',
    long_description='',
    url = 'https://github.com/romi/Segmentation',
    install_requires=install_requires,
    include_package_data=True,
)



