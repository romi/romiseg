# PyTorch

The `pyproject.toml` file specify the fixed following dependencies:

- `torch==2.5.1`
- `torchvision==0.20.1`

The reason is that it allows a wide range of CUDA Compute Capability, from `5.x` to `9.x`.

## Compatibility issues

If you have compatibility issues and wish to access older or newer packages with a specific CUDA version, go to the official [PyTorch](https://pytorch.org/get-started/previous-versions/) webpage.

## Manual installation

### CUDA 11.8
```shell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```
### CUDA 12.1
```shell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```
### CUDA 12.4
```shell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```
### CPU only
```shell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```
