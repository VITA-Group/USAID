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
    image[0] = image[0] + 0.3964
    image[1] = image[1] + 0.3695
    image[2] = image[2] + 0.2726
    return (image.numpy() * 255).transpose(1, 2, 0)

def demean(inputs):
    bgr_mean = np.array([0.4078705409, 0.4575245789, 0.48109378172]).reshape((3,1,1))
    inputs = inputs - bgr_mean  # inputs in [0,1]
    return inputs


class MultiDataSet(data.Dataset):
    """input and label image dataset"""

    def __init__(self, datadir = '../../VOCdevkit/VOC2012', cropSize=50, inSize=300, testFlag=False, preload=True):
        super(MultiDataSet, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        labelExtension: data = .jpg, label = .png
        """
        # self.root = root
        self.datadir = datadir
        self.cropSize = cropSize
        self.inSize = inSize
        # self.mean = np.array([0.3964, 0.3695, 0.2726])
        # self.fcn_mean = np.array([.485, .456, .406])
        # self.fcn_normal = np.array([.229, .224, .225])
        # self.fileDir = join(self.root, phase)
        # self.labelExtension = labelExtension
        self.preload = preload
        self.testFlag = testFlag
        self.mean = np.array([0.4078705409, 0.4575245789, 0.48109378172])
        if self.testFlag:
            split = 'val'
        else:
            split = 'train'
        self.imgsets_dir = join(datadir, 'ImageSets/Segmentation/%s.txt' % split)

        # self.image_filenames = [image_name for image_name in listdir(self.fileDir + '/Sat') if
        #                         is_image_file(image_name)]


        if self.preload:
            self.images = []
            self.labels = []
            self._pre_load()
        self.classdict = {0:'background', 1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car',
                          8:'cat', 9:'chair', 10:'cow', 11:'diningtable', 12:'dog', 13:'horse', 14:'motorbike',
                          15:'person', 16:'pottedplant', 17:'sheep', 18:'sofa', 19:'train', 20:'tvmonitor', 21:'void'}
        # self.classdict = {1: "urban", 2: "agriculture", 3: "rangeland", 4: "forest", 5: "water", 6: "barren",
        #                   0: "unknown"}

    def __getitem__(self, index):
        '''
        if not self.preload:
            timeStart = time.time()
            Satsample = cv2.imread(join(self.fileDir, 'Sat/' + self.image_filenames[index]))
            image = cv2.cvtColor(Satsample, cv2.COLOR_BGR2RGB)
            labelsamplename = find_label_map_name(self.image_filenames[index], self.labelExtension)
            #labelsample = cv2.imread(join(self.fileDir, 'Label/' + labelsamplename))
            #label = cv2.cvtColor(labelsample, cv2.COLOR_BGR2RGB)
            timeRead = time.time()
            #label = RGB_mapping_to_class(label)
            label = scipy.io.loadmat(join(self.fileDir, 'Notification/' +
                                          labelsamplename.replace('.png', '.mat')))["label"]     #???
            timeLabelTrans = time.time()
        else:
            image = self.images[index]
            label = self.labels[index]
        '''
        image = self.images[index]
        label = self.labels[index]

        image, label = self._transform(image, label)
        # image = image / 255.
        image = image / 255 - self.mean
        image = image.transpose(2, 0, 1)  # change to BRG
        # image = image/255 - self.fcn_mean
        # for i, normal in enumerate(self.fcn_normal):
        #     image[i, ...] = image[i, ...] / normal
        # timeFinish = time.time()
        # print("timeRead is %.2f \n timeLabelTrans is %.2f \n timeLabelTrans is %.2f \n" %
        #       (timeRead - timeStart, timeLabelTrans - timeRead, timeFinish - timeLabelTrans))
        return image.astype(np.float32), label.astype(np.int64)

    def _transform(self, image, label):
        # Scaling
        # scale_factor = random.uniform(1, 2)
        # scale = math.ceil(scale_factor * self.cropSize)
        #

        image = cv2.resize(
            image,
            (self.inSize, self.inSize),
            interpolation=cv2.INTER_LINEAR,
        )

        if not self.testFlag:

            label = cv2.resize(
                label,
                (self.inSize, self.inSize),
                interpolation=cv2.INTER_NEAREST,
            )

            '''
            # Crop
            h, w, _ = image.shape
            w_offset = random.randint(0, max(0, w - self.cropSize - 1))
            h_offset = random.randint(0, max(0, h - self.cropSize - 1))

            image = image[h_offset:h_offset + self.cropSize,
                          w_offset:w_offset + self.cropSize, :]
            label = label[h_offset:h_offset + self.cropSize,
                          w_offset:w_offset + self.cropSize]
            '''
            # Rotate
            rotate_time = np.random.randint(low=0, high=4)
            np.rot90(image, rotate_time)
            np.rot90(label, rotate_time)

            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW

        return image, label

    def _pre_load(self):
        print("preloading images and labels")
        with open(self.imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                Satsample = cv2.imread(join(self.datadir, "JPEGImages/%s.jpg" % name))
                image = cv2.cvtColor(Satsample, cv2.COLOR_BGR2RGB)
                labelsample = cv2.imread(join(self.datadir, "SegmentationClass/%s.png" % name))
                label = cv2.cvtColor(labelsample, cv2.COLOR_BGR2RGB)
                label = RGB_mapping_to_class(label)
                self.images.append(image.astype(np.float32))
                self.labels.append(label.astype(np.int64))
        print("finish preloading")
        '''
        for index, image_name in enumerate(self.image_filenames):
            print("loading image {}".format(index))
            Satsample = cv2.imread(join(self.fileDir, 'Sat/' + image_name))
            image = cv2.cvtColor(Satsample, cv2.COLOR_BGR2RGB)
            labelsamplename = find_label_map_name(image_name, self.labelExtension)
            labelsample = cv2.imread(join(self.fileDir, 'Label/' + labelsamplename))
            label = cv2.cvtColor(labelsample, cv2.COLOR_BGR2RGB)
            label = RGB_mapping_to_class(label)
            self.images.append(image.astype(np.float32))
            self.labels.append(label.astype(np.int64))
        print("finish preloading")
        '''
    def __len__(self):
        # return len(self.image_filenames)
        return len(self.images)


# dataset_train = MultiDataSet("/home/ckx9411sx/deepGlobe/land-train")
# np.save('/home/ckx9411sx/deepGlobe/temp/train_data_mc.npy', dataset_train)
# print(dataset_train)
