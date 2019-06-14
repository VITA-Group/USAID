# Segmentation-aware Image Denoising Without Knowing True Segmentation
Implement of the paper: <br> <br>
[Segmentation-aware Image Denoising Without Knowing True Segmentation](https://arxiv.org/abs/1905.08965) <br> <br>
Sicheng Wang, Bihan Wen, Junru Wu, Dacheng Tao, Zhangyang Wang <br>

## Overview
we propose a segmentation-aware image denoising model dubbed U-SAID, which does not need any ground-truth segmentation map in training, and thus can be applied to any image dataset directly. 
We demonstrate the U-SAID generates denoised image has:
* better visual quality <br>
* stronger robustness for subsequent semantic segmentation tasks <br>

We also manifest U-SAID's superior generalizability in three folds: 
* denoising unseen types of images <br>
* pre-processing unseen noisy images for segmentation <br>
* pre-processing unseen images for unseen high-level tasks. <br>

## Dataset
We use PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) for both training and validation.

## Training
Use USAID_train.py to train.

