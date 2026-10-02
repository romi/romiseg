# Welcome to ROMISeg

[![Licence](https://img.shields.io/github/license/romi/romiseg?color=lightgray)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fromi%2Fromiseg%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&logo=python&logoColor=white)]()
[![PyPI - Version](https://img.shields.io/pypi/v/romiseg?logo=pypi&logoColor=white)](https://pypi.org/project/romiseg/)
[![Conda - Version](https://img.shields.io/conda/vn/romi-eu/romiseg?logo=anaconda&logoColor=white&label=romi-eu&color=%2344A833)](https://anaconda.org/romi-eu/romiseg)
[![GitHub branch check runs](https://img.shields.io/github/check-runs/romi/romiseg/dev)](https://github.com/romi/romiseg)


![ROMI_ICON2_greenB.png](assets/images/ROMI_ICON2_greenB.png)

## Overview

ROMISeg is a specialized library for **semantic segmentation of plant images** as part of the ROMI (Robotics for Microfarms) pipeline.

It provides trained CNN models and utilities designed specifically for plant image analysis and segmentation.

Key features include:

- Pre-trained CNN models for plant image segmentation
- Support for multiple plant features (stem, flower, fruit, leaf, peduncle)
- Manual annotation tools for dataset creation
- Fine-tuning capabilities for custom plant species
- Integration with the ROMI pipeline

## Environment Setup

We strongly recommend using isolated environments to install ROMI libraries.
This documentation uses `conda` as both an environment and package manager.
If you don't have `miniconda3` installed, please refer to the [official documentation](https://docs.conda.io/en/latest/miniconda.html).

To create a new conda environment for ROMISeg:
```shell
conda create -n romiseg python==3.10
```

## Installation

### PyTorch & Dependencies

First, install PyTorch and its dependencies:
```shell
python -m pip install 'torch>=2.0.0' 'torchvision>=0.15.0' --extra-index-url 'https://download.pytorch.org/whl/cu118'
```

This installs PyTorch with CUDA 11.8 support, compatible with the latest PyTorch 2.x versions.

### Installing ROMISeg

You can install ROMISeg using either pip or by building from source:

#### Using pip:
```shell
python -m pip install romiseg
```

#### From source:
```shell
git clone https://github.com/romi/romiseg
cd romiseg
python -m pip install -e .
```

## Getting Started

- Learn about outputs and prerequisites: see [Segmentation2D](segmentation2d.md).
- Annotate images and adapt the model to your data: see [Annotation and Fine‑tuning](annotation-finetuning.md).

## Reference

The API reference is auto-generated from the Python sources under `src/romiseg`.
Navigate to "Reference API" in the sidebar for detailed documentation.
