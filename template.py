#!/usr/bin/env python3
"""
init_cadseg.py
Creates a ready-to-use project skeleton for CAD multi-label image segmentation (UNet++/ResNet34, SMP).
No heavy code—just structured placeholders, configs, and docs so you can start filling things in.

Usage:
  python init_cadseg.py --root cadseg        # default
  python init_cadseg.py --root myproj --overwrite
"""

import argparse
import json
import os
from pathlib import Path
from textwrap import dedent

# -----------------------------
# Template contents
# -----------------------------
GITIGNORE = dedent("""
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.venv/
env/
venv/
build/
dist/
*.egg-info/

# Data & runs
data/images/
data/masks/
data/splits/
runs/
outputs/
checkpoints/
.wandb/
mlruns/

# OS / IDE
.DS_Store
Thumbs.db
.idea/
.vscode/
""").lstrip()

README = dedent("""
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
""").lstrip()

REQUIREMENTS = dedent("""
torch>=2.2
torchvision>=0.17
segmentation-models-pytorch>=0.3.3
timm>=1.0.0
albumentations>=1.3.1
opencv-python-headless>=4.10.0.84
numpy>=1.26
scikit-image>=0.24
scikit-learn>=1.4
pyyaml>=6.0.1
tqdm>=4.66
rich>=13.7
matplotlib>=3.8
# optional
mlflow>=2.13
einops>=0.8
torchmetrics>=1.4
onnx>=1.16
onnxruntime>=1.18
pytest>=8.2
loguru>=0.7
""").strip()

# --- Configs (starter templates) ---
DATASET_YAML = dedent("""
# configs/dataset.yaml
root: ./data
images_dir: images
masks_dir: masks
classes:             # one channel per rule (order defines channel index)
  - rule_00_arrow_not_touching
  - rule_01_text_overlap_line
  - rule_02_missing_dimension
C: 3                  # must match number of classes
tile_size: 1024
tile_overlap: 256
num_workers: 4
seed: 42
folds: 1              # or use GroupKFold later
min_positive_area: 64 # px^2; used for positive tile mining
""").lstrip()

MODEL_YAML = dedent("""
# configs/model.yaml
arch: unet++            # UNet++ (nested U-Net)
encoder: resnet34
encoder_weights: imagenet
in_channels: 3
num_classes: 3          # keep in sync with dataset.yaml:C
dropout: 0.1
use_edge_aux: false     # optional edge-aware aux head (Sobel)
""").lstrip()

TRAIN_YAML = dedent("""
# configs/train.yaml
epochs: 40
batch_size: 4
optimizer: adamw
lr: 3.0e-4
weight_decay: 1.0e-4
sched: onecycle          # onecycle | cosine | reduce_on_plateau
amp: true
grad_clip: 1.0
freeze_encoder_stages: [0, 1]  # warmup freeze (stem + stage1)
unfreeze_at_epoch: 2
loss: dice_bce           # dice_bce | dice_focal | tversky | dice_ce
focal_gamma: 2.0
tversky_alpha: 0.7
tversky_beta: 0.3
class_weights: null      # e.g., [1.0, 1.4, 1.8] (inverse sqrt freq)
save_metric: macro_mIoU  # checkpoint by this metric
early_stop_patience: 8
""").lstrip()

AUG_YAML = dedent("""
# configs/aug.yaml
train:
  flip: true
  rotate90: true
  scale_jitter: 0.1          # ±10%
  brightness_contrast: 0.1
  gaussian_noise_std: 0.01
  gaussian_blur_ksize: 3     # set 0 to disable
  tiny_affine_px: 2          # translate ≤2px; no elastic to avoid bending lines
valid:
  resize: null               # or "fit_long_side: 2048"
  center_crop: null
infer:
  pad_to_tile: true
""").lstrip()

INFER_YAML = dedent("""
# configs/infer.yaml
tta: [hflip, vflip]         # optional
merge: mean                 # mean | gmean | max
tile_size: 1024
tile_overlap: 256
min_component_area:         # optional per-class small blob removal
  - 32
  - 32
  - 32
thresholds_file: configs/thresholds.json
""").lstrip()

THRESHOLDS_JSON = json.dumps({
    "rule_00_arrow_not_touching": 0.50,
    "rule_01_text_overlap_line": 0.50,
    "rule_02_missing_dimension": 0.50
}, indent=2)

CLASSES_JSON = json.dumps({
    "classes": [
        {"index": 0, "name": "rule_00_arrow_not_touching"},
        {"index": 1, "name": "rule_01_text_overlap_line"},
        {"index": 2, "name": "rule_02_missing_dimension"}
    ]
}, indent=2)

# Minimal placeholders for package modules (light docstrings, no logic)
PY_PLACEHOLDER = dedent('''\
"""
Placeholder module. Implement functions/classes as needed.
"""
''')

CONFIG_PY = dedent('''\
"""
Config loaders and typed dataclasses for dataset/model/train/aug/infer.
Fill with pydantic or dataclasses + YAML parsing.
"""
''')

DATASETS_PY = dedent('''\
"""
Dataset stubs:
- CADSegDataset: full images and masks (multi-label).
- TiledDataset: tile view over large canvases (with overlap).
- InferenceDataset: images only, for inference-time tiling.
"""
''')

TILING_PY = dedent('''\
"""
Tiling utilities:
- tile_image()
- stitch_probs() with weighted (e.g., Hann) blending
- valid-window masks
"""
''')

SAMPLING_PY = dedent('''\
"""
Samplers:
- ClassBalancedBatchSampler
- PositiveTileSampler
- WeightedRandomSampler for rare classes
"""
''')

TRANSFORMS_PY = dedent('''\
"""
Albumentations pipelines for train/valid/infer (CAD-safe).
Ensure mask and image transforms stay synchronized.
"""
''')

MASKS_PY = dedent('''\
"""
Mask IO and morphology utilities:
- load/save binary masks
- multi-label stacking (C channels)
- small-component removal, dilation/erosion (very mild)
- optional RLE helpers
"""
''')

COLLATE_PY = dedent('''\
"""
Custom collate functions for tile batches, dtype control, channel stacking.
"""
''')

BUILDER_PY = dedent('''\
"""
Model builder:
- UNet++ with ResNet-34 encoder (ImageNet)
- num_classes = C (multi-label, sigmoid used only at eval/metrics)
"""
''')

LOSSES_PY = dedent('''\
"""
Losses:
- Dice, BCEWithLogits, Focal, Tversky
- Per-class weighting wrappers and combos
"""
''')

METRICS_PY = dedent('''\
"""
Metrics:
- Per-class IoU/Dice, macro/micro averages
- Optional PR/ROC curve helpers (for calibration)
"""
''')

POSTPROCESS_PY = dedent('''\
"""
Post-processing:
- Per-class thresholds
- Small-blob removal
- Edge sharpening (optional)
"""
''')

EDGE_AUX_PY = dedent('''\
"""
Edge auxiliary head (optional):
- Sobel-based pseudo-targets for boundary sharpening
"""
''')

TRAIN_LOOP_PY = dedent('''\
"""
Training loop skeleton:
- AMP, grad clip, checkpointing, early stopping
- Encoder freeze/unfreeze schedule
- Class-balanced sampling
"""
''')

EVAL_LOOP_PY = dedent('''\
"""
Validation/Test loop skeleton:
- Full-image evaluation via tiling + stitching
- Per-class metrics aggregation and reporting
"""
''')

INFER_LOOP_PY = dedent('''\
"""
Inference loop skeleton:
- Folder or single image inference
- TTA and tile stitching
- Writes masks/overlays
"""
''')

OPTIMIZER_PY = dedent('''\
"""
Optimizer & scheduler builder:
- AdamW, OneCycle / Cosine / ReduceLROnPlateau
- Discriminative LR (encoder vs decoder)
"""
''')

HOOKS_PY = dedent('''\
"""
Callbacks/Hooks:
- Logging, LR finder, threshold calibration
- EMA (optional)
"""
''')

LOGGING_PY = dedent('''\
"""
Logging utilities using rich/loguru + tqdm progress.
"""
''')

SEED_PY = dedent('''\
"""
Seeding utilities for reproducibility (numpy, torch, albumentations).
"""
''')

CKPT_PY = dedent('''\
"""
Checkpoint helpers: save/load best model, resume, TorchScript/ONNX export helpers.
"""
''')

DIST_PY = dedent('''\
"""
DDP launcher/init (optional). Keep simple unless multi-GPU is required.
"""
''')

VIZ_PY = dedent('''\
"""
Quick visualizers for tiles, predictions, overlays, confusion matrices.
"""
''')

# CLI placeholders
PREPARE_DATASET_PY = dedent('''\
"""
CLI: prepare_dataset.py
- Scan data/, verify masks, compute class stats, build splits (GroupKFold by drawing).
"""
if __name__ == "__main__":
    print("prepare_dataset: stub")
''')

TRAIN_PY = dedent('''\
"""
CLI: train.py
- Load configs, build dataloaders/model, run training loop.
"""
if __name__ == "__main__":
    print("train: stub")
''')

VALIDATE_PY = dedent('''\
"""
CLI: validate.py
- Offline evaluation on valid/test; outputs per-class metrics.
"""
if __name__ == "__main__":
    print("validate: stub")
''')

CALIBRATE_PY = dedent('''\
"""
CLI: calibrate.py
- Sweep thresholds per class to maximize F1/IoU, save to thresholds.json.
"""
if __name__ == "__main__":
    print("calibrate: stub")
''')

INFER_PY = dedent('''\
"""
CLI: infer.py
- Batch inference → masks/overlays; optional polygon export.
"""
if __name__ == "__main__":
    print("infer: stub")
''')

EXPORT_PY = dedent('''\
"""
CLI: export.py
- Export to TorchScript/ONNX; optional INT8 PTQ (onnxruntime).
"""
if __name__ == "__main__":
    print("export: stub")
''')

# Scripts
VIS_SAMPLES_PY = dedent('''\
"""scripts/visualize_samples.py — quick grids of (image, GT, aug preview)."""
''')

SANITY_MASKS_PY = dedent('''\
"""scripts/sanity_check_masks.py — ensure masks align with images, values in {0,1}."""
''')

PROFILE_INFER_PY = dedent('''\
"""scripts/profile_inference.py — measure throughput/latency for different tile sizes/overlaps."""
''')

# Tests
TEST_TILING = dedent('''\
def test_tiling_placeholder():
    assert True
''')

TEST_TRANSFORMS = dedent('''\
def test_transforms_placeholder():
    assert True
''')

TEST_METRICS = dedent('''\
def test_metrics_placeholder():
    assert True
''')

# -----------------------------
# Files to create
# -----------------------------
FILES = {
    ".gitignore": GITIGNORE,
    "README.md": README,
    "requirements.txt": REQUIREMENTS,

    # configs
    "configs/dataset.yaml": DATASET_YAML,
    "configs/model.yaml": MODEL_YAML,
    "configs/train.yaml": TRAIN_YAML,
    "configs/aug.yaml": AUG_YAML,
    "configs/infer.yaml": INFER_YAML,
    "configs/thresholds.json": THRESHOLDS_JSON,

    # data
    "data/meta/classes.json": CLASSES_JSON,
    "data/splits/train.txt": "",
    "data/splits/valid.txt": "",
    "data/splits/test.txt": "",
    "data/masks/.keep": "",
    "data/masks/rule_00/.keep": "",
    "data/masks/rule_01/.keep": "",
    "data/masks/rule_02/.keep": "",
    "data/images/.keep": "",

    # package roots
    "cadseg/__init__.py": "",
    "cadseg/config.py": CONFIG_PY,

    # dataio
    "cadseg/dataio/__init__.py": "",
    "cadseg/dataio/datasets.py": DATASETS_PY,
    "cadseg/dataio/tiling.py": TILING_PY,
    "cadseg/dataio/sampling.py": SAMPLING_PY,
    "cadseg/dataio/transforms.py": TRANSFORMS_PY,
    "cadseg/dataio/masks.py": MASKS_PY,
    "cadseg/dataio/collate.py": COLLATE_PY,

    # models
    "cadseg/models/__init__.py": "",
    "cadseg/models/builder.py": BUILDER_PY,
    "cadseg/models/losses.py": LOSSES_PY,
    "cadseg/models/metrics.py": METRICS_PY,
    "cadseg/models/postprocess.py": POSTPROCESS_PY,
    "cadseg/models/edge_aux.py": EDGE_AUX_PY,

    # engine
    "cadseg/engine/__init__.py": "",
    "cadseg/engine/train_loop.py": TRAIN_LOOP_PY,
    "cadseg/engine/eval_loop.py": EVAL_LOOP_PY,
    "cadseg/engine/infer_loop.py": INFER_LOOP_PY,
    "cadseg/engine/optimizer.py": OPTIMIZER_PY,
    "cadseg/engine/hooks.py": HOOKS_PY,

    # utils
    "cadseg/utils/__init__.py": "",
    "cadseg/utils/logging.py": LOGGING_PY,
    "cadseg/utils/seed.py": SEED_PY,
    "cadseg/utils/ckpt.py": CKPT_PY,
    "cadseg/utils/distributed.py": DIST_PY,
    "cadseg/utils/viz.py": VIZ_PY,

    # CLI
    "cadseg/cli/__init__.py": "",
    "cadseg/cli/prepare_dataset.py": PREPARE_DATASET_PY,
    "cadseg/cli/train.py": TRAIN_PY,
    "cadseg/cli/validate.py": VALIDATE_PY,
    "cadseg/cli/calibrate.py": CALIBRATE_PY,
    "cadseg/cli/infer.py": INFER_PY,
    "cadseg/cli/export.py": EXPORT_PY,

    # scripts
    "scripts/visualize_samples.py": VIS_SAMPLES_PY,
    "scripts/sanity_check_masks.py": SANITY_MASKS_PY,
    "scripts/profile_inference.py": PROFILE_INFER_PY,

    # tests
    "tests/test_tiling.py": TEST_TILING,
    "tests/test_transforms.py": TEST_TRANSFORMS,
    "tests/test_metrics.py": TEST_METRICS,

    # notebooks
    "notebooks/.keep": "",
}

# -----------------------------
# Helpers
# -----------------------------
def write_file(base: Path, rel: str, content: str, overwrite: bool = False):
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"SKIP  {rel} (exists)")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"WRITE {rel}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="cadseg", help="Project root directory to create.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Creating project at: {root}")

    for rel, content in FILES.items():
        write_file(root, rel, content, overwrite=args.overwrite)

    print("\nDone. Next steps:")
    print(f"  1) cd {root}")
    print("  2) python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\\Scripts\\activate)")
    print("  3) pip install -r requirements.txt")
    print("  4) Edit configs/*.yaml (paths/classes).")
    print("  5) Start filling modules in cadseg/, then run: python -m cadseg.cli.train")

if __name__ == "__main__":
    main()
