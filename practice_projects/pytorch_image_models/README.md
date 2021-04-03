# Pytorch Image models

- [Introduction](#introduction)
- [Activities](#recent-activities)
- [Models](#models)
- [Features](#features)
- [Results](#results)
- [Train, Validation, Inference Scripts](#train-validation-inference-scripts)
- [Awesome Pytorch Resources](#awesome-pytorch-resources)
- [Licenses](#licenses)
- [Citing](#citing)

## Introduction

Py**T**orch **Im**age **M**odels (`timm`) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

## Models

All model architecture families include variants with pretrained weights. Some [training hparams](https://rwightman.github.io/pytorch-image-models/training_hparam_examples) to get started.

A full version of the list below with source link can be found in the [documentation](https://rwightman.github.io/pytorch-image-models/models/).

- Big Transfer ResNetV2 (BiT) - https://arxiv.org/abs/1912.11370
- CspNet (Cross-Stage Partial Networks) - https://arxiv.org/abs/1911.11929
- DeiT (Vision Transformer) - https://arxiv.org/abs/2012.12877
- DenseNet - https://arxiv.org/abs/1608.06993
- DLA - https://arxiv.org/abs/1707.06484

- DPN (Dual-Path Network) - https://arxiv.org/abs/1707.01629

- EfficientNet (MBConvNet Family)
  - EfficientNet NoisyStudent (B0-B7, L2) - https://arxiv.org/abs/1911.04252
  - EfficientNet AdvProp (B0-B8) - https://arxiv.org/abs/1911.09665

## Activities

### April 3, 2021

- Created the Readme file
