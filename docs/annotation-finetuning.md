# Annotation and Fine‑tuning

This guide explains how to annotate images and fine‑tune a pixel‑wise segmentation network on your own dataset.

Use cases:

- Improve ROMI pipeline segmentation on plant species other than _Arabidopsis thaliana_.
- Adapt to different imaging setups or conditions.

Example: 
A tomato scan segmented with an Arabidopsis‑trained network performs poorly.
By annotating a few tomato images and fine‑tuning, results improve enough for reliable 3D reconstruction.

## Prerequisites

- A ROMI scan with the `Colmap` task completed.
- `LabelMe` installed for manual annotations.
- A configuration TOML file compatible with your pipeline.

## Quick start
Run the tool with your config file:
```bash
finetune --config /path/to/config_file
```

## Configuration
The tool relies on the same TOML file used by the virtual scanner. Update the Segmentation2D section as needed (example):
```toml
[Segmentation2D]
upstream_task = "Scan"
query = "{\"channel\":\"rgb\"}"
labels = "background,flower,peduncle,stem,bud,leaf,fruit"
model_name = "Resnet101"
model_segmentation_name = "Resnet101_896_896_epoch51.pt" # model to fine‑tune
Sx = 896
Sy = 896
learning_rate = 0.0001

[Finetune]
finetune_epochs = 10
batch = 1
```
- If the pre‑trained network is missing, it is fetched from db.romi-project.eu.
- Adjust the number of epochs and batch size as appropriate.

## Workflow
1. Select 2–3 images to annotate from your scan.
2. `LabelMe` opens on the first image. Use Create Polygon to annotate: flower, stem, leaf, fruit, peduncle.
3. Save the .json file in the suggested folder (this is the directory_images folder; the selected images are also copied there).
4. `LabelMe` will open the next selected image automatically. Repeat.
5. When all images are labeled, a pop‑up previews a random sample of the training dataset. Close it and press Enter to continue.
6. Training starts automatically for the number of epochs set in the config file.
7. When training finishes, weights are saved and the config is updated with the new fine‑tuned model name.
8. Re‑run the ROMI pipeline on the scan; segmentation and reconstruction should improve.

## Notes

For more information about training, notably directory layout and configuration files, read [Training](training.md).