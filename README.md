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
pip install numpy matplotlib pillow jupyter ipykernel tqdm scikit-learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python==4.12.0.88
pip install imagecodecs
```

## Contact
- Ousseini Hamza Abdoul Djalil - Engineer (abdouldjalilo@gmail.com)
- Rousseau David - Univ. Professor