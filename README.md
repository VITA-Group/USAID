# Segmentation-aware Image Denoising Without Knowing True Segmentation
Implement of the paper: <br> <br>
[Segmentation-aware Image Denoising Without Knowing True Segmentation](https://arxiv.org/abs/1905.08965) <br> <br>
Sicheng Wang, Bihan Wen, Junru Wu, Dacheng Tao, Zhangyang Wang <br>

## Overview
we propose a segmentation-aware image denoising model dubbed **U-SAID**, which does not need any ground-truth segmentation map in training, and thus can be applied to any image dataset directly. 
We demonstrate the U-SAID generates denoised image has:
* better visual quality; <br>
* stronger robustness for subsequent semantic segmentation tasks. <br>

We also manifest U-SAID's superior generalizability in three folds: 
* denoising unseen types of images; <br>
* pre-processing unseen noisy images for segmentation; <br>
* pre-processing unseen images for unseen high-level tasks. <br>

## Methods
![](https://github.com/sharonwang1/seg_denoising/blob/master/docs/images/FlowChart.png)
<p align="center">
<b>U-SAID</b>: Network architecture. The USA module is composed of a feature embedding sub-network for transforming the denoised image to a feature space, followed by an unsupervised segmentation sub-network that projects the feature to a segmentation map and calculates its pixel-wise uncertainty.
</p>

## Visual Examples
### Visual comparison on [Kodak](http://r0k.us/graphics/kodak/) Images
![](https://github.com/sharonwang1/seg_denoising/blob/master/docs/images/kodak_ship.jpg)

### Semantic segmentation from [Pascal VOC 2012 validation set](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
![](https://github.com/sharonwang1/seg_denoising/blob/master/docs/images/VOC_segmentation.jpg)

## How to run
### Dependences
* [PyTorch](http://pytorch.org/)(<0.4)
* [torchvision](https://github.com/pytorch/vision)
* OpenCV for Python
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (TensorBoard for PyTorch)

## Citation
If you use this code for your research, please cite our paper.
```
@misc{1905.08965,
Author = {Sicheng Wang and Bihan Wen and Junru Wu and Dacheng Tao and Zhangyang Wang},
Title = {Segmentation-Aware Image Denoising without Knowing True Segmentation},
Year = {2019},
Eprint = {arXiv:1905.08965},
}
```
