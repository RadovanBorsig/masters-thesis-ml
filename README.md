# Neural Networks in Ophthalmology

This repository contains the implementation of my Master's thesis, focused on the classification of retinal Optical Coherence Tomography (OCT) images using deep learning.

The project compares different neural network architectures for classifying OCT images into four diagnostic categories:
- CNV
- DME
- DRUSEN
- NORMAL

## Technologies

- Python
- PyTorch
- torchvision
- scikit-learn
- timm
- OpenCV
- matplotlib
- Optuna

## Project Overview

The project covers the following areas:

- Image preprocessing and normalization
- Data augmentation
- Training and evaluation of deep learning models
- Comparison of different CNN and Transformer-based architectures
- Hyperparameter optimization using Optuna
- Model evaluation using standard classification metrics
- Model interpretability using Grad-CAM and Attention Rollout

## Models

The project evaluates several CNN and Transformer-based architectures, including:

- ResNet50
- DenseNet121
- VGG16
- InceptionV3
- ConvNeXt
- EfficientNetB3
- Xception
- ViT Dino
- DeiT
- Swin Transformer
