#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from setuptools import find_packages
from setuptools import setup

dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

install_requires = [
    'appdirs',
    'future',
    'labelme',
    'mako',
    'Pillow',
    'pyqt5==5.14',
    'pyyaml',
    'requests',
    'tensorboard',
    'torch<1.11',
    'torchvision',
    'tqdm'
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
    url='https://github.com/romi/romiseg',
    install_requires=install_requires,
    include_package_data=True,
)
