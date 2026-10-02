# [![ROMI_logo](docs/assets/images/ROMI_logo_green_25.svg)](https://romi-project.eu) / romiseg

[![Licence](https://img.shields.io/github/license/romi/romiseg?color=lightgray)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fromi%2Fromiseg%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&logo=python&logoColor=white)]()
[![PyPI - Version](https://img.shields.io/pypi/v/romiseg?logo=pypi&logoColor=white)](https://pypi.org/project/romiseg/)
[![Conda - Version](https://img.shields.io/conda/vn/romi-eu/romiseg?logo=anaconda&logoColor=white&label=romi-eu&color=%2344A833)](https://anaconda.org/romi-eu/romiseg)
[![GitHub branch check runs](https://img.shields.io/github/check-runs/romi/romiseg/dev)](https://github.com/romi/romiseg)

## Overview

This package contains trained CNN models and methods dedicated to performs semantic segmentation of plant images.

## Installation

### Clone the sources

```shell
git clone https://github.com/romi/romiseg
```

### Optional - Create a conda environment

```shell
conda create -n romiseg python==3.10
```

### Install PyTorch & dependencies

```shell
python -m pip install 'torch>=2.0.0' 'torchvision>=0.15.0' --extra-index-url 'https://download.pytorch.org/whl/cu118'
```

We install it for CUDA 11.8, which is compatible with the latest PyTorch 2.x versions.

### Install `romiseg`

```shell
cd romiseg
python -m pip install -e .
```

## Usage and Documentation

- Segmentation2D module: outputs, prerequisites, and details are documented at docs/segmentation2d.md
- Annotation and fine‑tuning: step‑by‑step guide at docs/annotation-finetuning.md

For the full documentation website, see https://romi.github.io/romiseg/
