# HeteroCD-GOLD

A heterogeneous remote sensing image change detection framework based on **G**uided **O**nline **L**earning for **D**istillation.

## Overview

This project implements GOLD, a three-branch fully-supervised online distillation model for detecting changes between optical and SAR remote sensing images. The model uses online knowledge distillation from homogeneous optical-optical pairs (teacher branch) to guide heterogeneous optical-SAR change detection (student branch), fundamentally addressing cross-modal feature space differences.

## Key Features

- **Three-branch Architecture**: Homogeneous teacher branch (optical-optical) and heterogeneous student branch (optical-SAR) with shared temporal-1 optical encoder
- **Online Knowledge Distillation**: Real-time transfer of high-level change features and quality label information during each iteration
- **Difference Map Attention Transfer**: Spatial and channel attention mechanisms with saliency maps for enhanced change perception
- **Dynamic Weight Allocation**: Uncertainty-based adaptive loss weighting for change detection, distillation, and attention losses
- **LabelmeCD-AI Annotation Tool**: Synchronized dual-image display with AI-driven pre-annotations

## Dataset
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/Mercyiris/remote-sensing-change-detection)
[![ModelScope](https://img.shields.io/badge/ModelScope-Dataset-blue)](https://modelscope.cn/datasets/Mriris/remote-sensing-change-detection)

First benchmark dataset combining optical-optical and optical-SAR time-series pairs:
- **Gaofen-2** high-resolution optical images
- **Gaofen-3** SAR images  
- **Sentinel-2** multispectral images

## Requirements

```bash
# intsall PyTorch
uv pip install torch torchvision

# install other dependencies
uv sync # or use uv pip install -r requirements.txt
```

## Dataset Structure

```
data/
├── train/
│   ├── A/          # Optical images (time 1)
│   ├── B/          # SAR images (time 2)
│   ├── C/          # Optical images (time 2)
│   └── Label/      # Change labels
└── val/
    ├── A/
    ├── B/
    ├── C/
    └── Label/
```

## Data Preprocessing

Process raw remote sensing images into training-ready patches:

```bash
python datasets/preprocess.py
```

**Key Features:**
- Geographic coordinate-based overlap detection (80% threshold)
- Pure black tile filtering (95% threshold)
- Automatic train/val/test split (80%/20%/20%)
- 512×512 patch generation with optional data augmentation

**Input Format:** `{basename}_{A|B|D|E}.{tif|png}` where A/B/D are multi-temporal images and E is the change label.

## Quick Start

### Training

```bash
python train.py
```

### Testing

```bash
python predict.py
```

## Repository Structure
```

├── models/                 
│   ├── hetecd.py           # Baseline backbone
│   ├── resnet.py           # ResNet backbones
│   ├── resnetCD.py         # ResNet-based change head
│   ├── BiSRNet.py          # Baseline model
│   └── base_block.py       # Common blocks
├── datasets/               
│   ├── RS_ST.py            # Dataset loader
│   ├── preprocess.py       # Patch generation & split
├── utils/                  # Training, loss, and evaluation tools
│   ├── utils_fit.py        # Training loop
│   ├── loss.py             # Loss definitions
│   ├── eval.py             # Metrics & evaluation
│   └── transform.py        # Data transforms
├── train.py                # Training entry
├── predict.py              # Inference and visualization
├── requirements.txt        # Python dependencies
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{heterocd_gold,
  title={GOLD: Guided Online Learning for Distillation for Heterogeneous Remote Sensing Image Change Detection},
  author={Tingxuan Yan},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2025}
}
```