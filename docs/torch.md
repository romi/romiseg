# PyTorch

## Fixed dependencies (recommended for reproducible builds)

The `pyproject.toml` file specify the fixed following dependencies:

- `torch==2.5.1`
- `torchvision==0.20.1`

The reason is that it allows a wide range of CUDA Compute Capability, from `5.x` to `9.x`.

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

## Verify Compatibility

After installation, run the short script below to make sure your GPU is supported and that the installed CUDA toolchain matches your hardware:

```python
import torch

# List of compute capabilities the current wheel supports
print("Supported CUDA architectures:", torch.cuda.get_arch_list())

# Properties of the first detected GPU (device 0)
device = torch.cuda.get_device_properties(0)
print(f"Detected GPU: {device.name} (sm_{device.major}{device.minor})")
```

**How to interpret the output**

| Situation                                                      | What to do                                                                                                                                    |
|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `sm_xx` **appears** in `torch.cuda.get_arch_list()`            | ✅ Your GPU is compatible – you can start training.                                                                                            |
| `sm_xx` **does NOT appear**                                    | ❌ The wheel you installed does **not** support your GPU. Choose a different CUDA version (_e.g._, a lower‑CUDA wheel) or install from source. |
| No CUDA device is found (`torch.cuda.is_available() == False`) | ❌ You may be on a CPU‑only machine or need to install the **CPU‑only** wheel.                                                                 |

> **Important**: You should match the major version from the supported architecture list to the major or the device architecture.
> For example, if your device is _sm_52_ and the `get_arch_list` returns
`['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']` it will work.
> It will not work if it returns `['sm_75', 'sm_80', 'sm_86', 'sm_90']`.

## When you need a different version

If you run into compatibility problems (_e.g._, older GPU, newer CUDA toolkit not yet covered), you can:

1. **Browse the official list** of previous PyTorch versions and their CUDA wheels on the [Previous Versions page](https://pytorch.org/get-started/previous-versions/).
2. Pick the exact version you need (_e.g._, `torch==2.10.0` for CUDA 12.6) and adjust the `pip` command accordingly.
3. Optionally, pin the version in `pyproject.toml`  to guarantee reproducibility.

