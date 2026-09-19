# Federated Self-Supervised Industrial Anomaly Detection under Domain Shift

This repository contains the implementation developed for my MSc thesis:

**Federated Self-Supervised Industrial Anomaly Detection under Domain Shift**

## Overview

The project investigates industrial visual anomaly detection in a federated
learning setting using the MVTec AD Bottle category.

The framework combines:

- ImageNet-pretrained ResNet18 with a frozen backbone
- Patch-level features from ResNet18 layer2 and layer3
- 384-dimensional patch representations
- Lightweight residual adapter: 384 → 128 → 384
- Self-supervised feature denoising using normal samples
- Client-specific normal feature memory banks
- Federated Averaging (FedAvg)
- FedProx under controlled Non-IID domain shift
- Image-level and pixel-level AUROC evaluation

## Dataset

The experiments use the **MVTec Anomaly Detection (MVTec AD)** dataset,
restricted to the Bottle category.

The dataset itself is not included in this repository and should be obtained
from the official MVTec AD source.

## Experimental Setup

- Training images: 209 normal images
- Test images: 83
- Normal test images: 20
- Anomalous test images: 63
- Federated clients: 3
- Communication rounds: 10
- Backbone: ResNet18
- Residual adapter: 384 → 128 → 384
- Trainable adapter parameters: 98,816

## Main Results

The federated configuration achieved approximately:

- Image-level AUROC: 0.998
- Pixel-level AUROC: 0.982

The results indicate that strong anomaly-detection and localization performance
can be preserved while federating only a lightweight residual adapter and
keeping raw client data and normal memory banks local.

## Implementation

The complete experimental implementation is provided in:

`federated_industrial_anomaly_detection.ipynb`

The experiments were developed using Python and PyTorch in a Kaggle environment
with an NVIDIA Tesla T4 GPU.

## Author

Charan Siddarth Kovilpatti Gajendran

MSc Data Science, AI and Digital Innovation
