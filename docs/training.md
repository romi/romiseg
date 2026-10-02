# Training a network from a database.

## Recommended directory layout
Use the following structure for training, weights, and TensorBoard logs:
```
.
├── weights           # model weights
├── board             # TensorBoard logs
└── data
    └── dataset
        ├── train
        │   ├── images   # RGB images (png/jpg)
        │   └── labels   # label masks
        ├── val
        │   ├── images
        │   └── labels
        └── test
            ├── images
            └── labels
```

Reference these directories in your config:
```toml
[TrainingDirectory]
path = "/home/you/shared/"
directory_weights = "weights/"
tsboard = "board/"
directory_dataset = "dataset_stem_green"

[Segmentation2D]
upstream_task = "Scan"
query = "{\"channel\":\"rgb\"}"
labels = "background,flower,peduncle,stem,bud,leaf,fruit"
model_name = "Resnet101"
model_segmentation_name = "Resnet101_896_896_epoch51.pt"
Sx = 896
Sy = 896
epochs = 5
batch = 1
learning_rate = 0.0001
```

## Generate a dataset with blender

To be completed

## Parameters

the training parameters are to be filled in `parameters_train.toml`.

- size of the training images (center crop from images in train/images)
- type of segmentation network (for now they are sourced in [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch)
- batch size
- number of epochs
- learning rate

## Training a network
You can train via CLI:
```bash
train_cnn --config /path/to/config
```
Or import the training function in Python for experiments.

## Notes
- The base network architectures are sourced from segmentation_models_pytorch.
- Ensure your label masks follow the expected format (see dataset utilities in the codebase).
- CUDA >=11.8 and PyTorch 2.x are recommended as in the installation guide.
