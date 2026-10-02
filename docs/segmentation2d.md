# Segmentation2D

The `Segmentation2D` module generates per-class prediction masks for plant images within the ROMI pipeline.

Requirements:

- The `Colmap` task must be completed.
- Preferably use `Undistorted` images.

It outputs six masks per input image for the following classes:

- background
- stem
- flower
- fruit
- leaf
- peduncle

These predictions come from a CNN trained mainly on virtual Arabidopsis images generated with _ROMI's blender virtual scanner_.
You can improve segmentation on your own species and imaging conditions using the annotation and fine‑tuning workflow.

See also: [Annotation and Fine‑tuning](annotation-finetuning.md).
