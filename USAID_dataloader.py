# Prepare Dataset
from os import listdir
from os.path import join
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
import cv2
from sklearn.feature_extraction.image import extract_patches_2d
import math
import time
import scipy.io

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def find_label_map_name(img_filenames, labelExtension=".png"):
    img_filenames = img_filenames.replace('_sat.jpg', '_mask')
    return img_filenames + labelExtension


def RGB_mapping_to_class(label):
    l, w = label.shape[0], label.shape[1]
    classmap = np.zeros(shape=(l, w))
    indices = np.where(np.all(label == (0, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 0
    indices = np.where(np.all(label == (128, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 1
    indices = np.where(np.all(label == (0, 128, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 2
    indices = np.where(np.all(label == (128, 128, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 3
    indices = np.where(np.all(label == (0, 0, 128), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 4
    indices = np.where(np.all(label == (128, 0, 128), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 5
    indices = np.where(np.all(label == (0, 128, 128), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 6
    indices = np.where(np.all(label == (128, 128, 128), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 7
    indices = np.where(np.all(label == (64, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 8
    indices = np.where(np.all(label == (192, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 9
    indices = np.where(np.all(label == (64, 128, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 10
    indices = np.where(np.all(label == (192, 128, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 11
    indices = np.where(np.all(label == (64, 0, 128), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 12
    indices = np.where(np.all(label == (192, 0, 128), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 13
    indices = np.where(np.all(label == (64, 128, 128), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 14
    indices = np.where(np.all(label == (192, 128, 128), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 15
    indices = np.where(np.all(label == (0, 64, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 16
    indices = np.where(np.all(label == (128, 64, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 17
    indices = np.where(np.all(label == (0, 192, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 18
    indices = np.where(np.all(label == (128, 192, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 19
    indices = np.where(np.all(label == (0, 64, 128), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 20
    indices = np.where(np.all(label == (255, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 21

    #     plt.imshow(colmap)
    #     plt.show()
    return classmap


def classToRGB(label):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3))
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [128, 0, 0]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 128, 0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [128, 128, 0]
    indices = np.where(label == 4)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 128]
    indices = np.where(label == 5)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [128, 0, 128]
    indices = np.where(label == 6)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 128, 128]
    indices = np.where(label == 7)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [128, 128, 128]
    indices = np.where(label == 8)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [64, 0, 0]
    indices = np.where(label == 9)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [192, 0, 0]
    indices = np.where(label == 10)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [64, 128, 0]
    indices = np.where(label == 11)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [192, 128, 0]
    indices = np.where(label == 12)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [64, 0, 128]
    indices = np.where(label == 13)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [192, 0, 128]
    indices = np.where(label == 14)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [64, 128, 128]
    indices = np.where(label == 15)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [192, 128, 128]
    indices = np.where(label == 16)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 64, 0]
    indices = np.where(label == 17)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [128, 64, 0]
    indices = np.where(label == 18)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 192, 0]
    indices = np.where(label == 19)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [128, 192, 0]
    indices = np.where(label == 20)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 64, 128]
    indices = np.where(label == 21)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 255]

    # transform = ToTensor();
    #     plt.imshow(colmap)
    #     plt.show()
    return colmap.astype(np.uint8)


def class_to_target(inputs, numClass):
    batchSize, l, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    target = np.zeros(shape=(batchSize, l, w, numClass), dtype=np.float32)
    for index in range(21):
        indices = np.where(inputs == index)
        temp = np.zeros(shape=21, dtype=np.float32)
        temp[index] = 1
        target[indices[0].tolist(), indices[1].tolist(), indices[2].tolist(), :] = temp
    return target.transpose(0, 3, 1, 2)


def label_bluring(inputs):
    batchSize, numClass, height, width = inputs.shape
    outputs = np.ones((batchSize, numClass, height, width), dtype=np.float)
    for batchCnt in range(batchSize):
        for index in range(numClass):
            outputs[batchCnt, index, ...] = cv2.GaussianBlur(inputs[batchCnt, index, ...].astype(np.float), (7, 7), 0)
    return outputs

def inputImgTransBack(inputs):
    image = inputs[0].to("cpu")
    image[0] = image[0] + 0.48109378172
    image[1] = image[1] + 0.4575245789
    image[2] = image[2] + 0.4078705409
    return (image.numpy() * 255).transpose(1, 2, 0)

def bgr_demean(inputs):  # inputs in [0,1]
    bgr_mean = np.array([0.4078705409, 0.4575245789, 0.48109378172]).reshape((3,1,1))   #BGR
    return inputs - bgr_mean

def rgb_demean(inputs):
    rgb_mean = np.array([0.48109378172, 0.4575245789, 0.4078705409]).reshape((3, 1, 1))
    inputs = inputs - rgb_mean  # inputs in [0,1]
    return inputs

class MultiDataSet(data.Dataset):
    """input and label image dataset"""

    def __init__(self, datadir = '/VOCdevkit/VOC2012', cropSize=48, testFlag=False, Scale=False):
        super(MultiDataSet, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        labelExtension: data = .jpg, label = .png
        """
        self.datadir = datadir
        self.cropSize = cropSize
        self.Scale = Scale
        self.testFlag = testFlag
        if self.testFlag:
            split = 'val'
        else:
            split = 'train'
        self.imgsets_dir = join(datadir, 'ImageSets/Segmentation/%s.txt' % split)

        self.images = []
        self._pre_load()
        self.classdict = {0:'background', 1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car',
                          8:'cat', 9:'chair', 10:'cow', 11:'diningtable', 12:'dog', 13:'horse', 14:'motorbike',
                          15:'person', 16:'pottedplant', 17:'sheep', 18:'sofa', 19:'train', 20:'tvmonitor', 21:'void'}

    def __getitem__(self, index):
        image = self.images[index]
        image = self._transform(image)
        image = image / 255.
        image = image.transpose(2, 0, 1)  #  H,W,C --> C,H,W
        return image.astype(np.float32)


    def _transform(self, image):
        if not self.testFlag:
            # Rotate
            rotate_time = np.random.randint(low=0, high=4)
            np.rot90(image, rotate_time)

            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()

        return image

    def _pre_load(self):
        # print("preloading images and labels")
        print("preloading images")
        with open(self.imgsets_dir) as imgset_file:
            for name in imgset_file:
                # print(name)
                name = name.strip()
                Satsample = cv2.imread(join(self.datadir, "JPEGImages/%s.jpg" % name))
                img = cv2.cvtColor(Satsample, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape

                ####### extract patch + scale ###########
                if not self.testFlag:
                    if self.Scale:
                        extract_patches = extract_patches_2d(img, (int(0.8*h), int(0.8*w)), max_patches=0.02)
                        for k in range(len(extract_patches)):
                            patches = cv2.resize(extract_patches[k], (self.cropSize, self.cropSize),
                                                 interpolation=cv2.INTER_CUBIC)
                            self.images.append(patches.astype(np.float32))
                    else:
                        patches = extract_patches_2d(img, (self.cropSize, self.cropSize), max_patches=0.001)
                        self.images.append(patches.astype(np.float32))
                else:
                    self.images.append(img.astype(np.float32))

        if not self.Scale:
            if not self.testFlag:
                self.images = np.concatenate(self.images)


        print("finish preloading")

    def __len__(self):
        return len(self.images)
