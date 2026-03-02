# Image Classification Training – CNN with PyTorch

## Overview

This repository contains a complete educational pipeline for image classification using classical CNN architectures (ResNet, VGG) with PyTorch.

The training session covers:

- Data preparation
- Transfer learning
- Training & validation pipeline
- Early stopping
- Model checkpointing
- Confusion matrix
- Grad-CAM visualization

---

## Dataset Structure

The dataset must follow this structure:

dataset/
    healthy/
    diseased/

Each folder represents a class.

---

## Installation

```bash
conda create -n cnn_training python=3.10
conda activate cnn_training
pip install torch torchvision scikit-learn matplotlib opencv-python
```

## Contact
- Ousseini Hamza Abdoul Djalil - Engineer (abdouldjalilo@gmail.com)
- Rousseau David - Univ. Professor