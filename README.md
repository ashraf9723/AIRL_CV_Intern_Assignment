# Project README

This repository contains two Colab notebooks:
1. **Q1** – Vision Transformer on CIFAR-10  
2. **Q2** – Text-Driven Image & Video Segmentation  

## 📌 Q1 – Vision Transformer on CIFAR-10

### Overview
This notebook implements a **Vision Transformer (ViT)** from scratch and trains it on the **CIFAR-10 dataset**.  
The architecture includes patch embedding, multi-head self-attention, MLP layers, and stochastic depth for regularization.

### Features
- Custom implementations of:
  - Patch Embedding
  - Multi-Head Self-Attention
  - Transformer Encoder Block
  - Vision Transformer classifier
- Data augmentation with AutoAugment and normalization
- CIFAR-10 dataset loaders for training/testing
- Training loop with:
  - Warm-up learning rate scheduling (first 5 epochs)
  - Cosine Annealing scheduler
- Evaluation after each epoch

### Results
- Achieved **~84.95% accuracy** on CIFAR-10 test set after 100 epochs.

### How to Run
1. Install dependencies:
   ```bash
   pip install torch torchvision tqdm
