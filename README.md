# CADSeg — Multi-label CAD Violation Segmentation

This repository is a **template** for training a **multi-label** image segmentation model
(one channel per CAD rule violation) on high-resolution CAD drawings using tiling.

## Key ideas
- **Architecture**: UNet++ with ResNet-34 encoder (ImageNet-pretrained) via `segmentation_models_pytorch`.
- **Output**: C channels (one per rule), **sigmoid** activations (multi-label).
- **Loss**: Per-class Dice + BCEWithLogits (class-weighted).
- **Tiling**: Large canvases → 1024×1024 tiles with ~256 px overlap and weighted stitching.
- **Sampling**: Class-balanced batch sampler that prefers tiles with positives.
- **Thresholds**: Calibrate per-class thresholds on validation set.

## Quick start
1) Create a venv and install `requirements.txt`.
2) Fill `configs/*.yaml` with your paths/classes.
3) Add your data under `data/` (images + masks per rule).
4) Implement loaders and loops under `cadseg/` (placeholders provided).
5) Train via `cadseg/cli/train.py` (entry point stub provided).

> This template intentionally ships **without** heavy code so you can adapt quickly to your setup.
