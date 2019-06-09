import matlab.engine
eng = matlab.engine.start_matlab()
import math
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from skimage.measure import compare_psnr, compare_ssim


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, data_range):
    img = torch.transpose(img, 1, 3)
    imclean = torch.transpose(imclean, 1, 3)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range, multichannel=True)
    return (SSIM/Img.shape[0])

def batch_NIQE(img):
    # eng = matlab.engine.start_matlab()
    img_ = torch.transpose(img, 1, 3).data.cpu().numpy().astype(np.float32)
    NIQE = 0
    for i in range(img_.shape[0]):
        NIQE += eng.niqe(matlab.double(img_[i,:,:,:].tolist()))
    # eng.quit()
    return (NIQE/img_.shape[0])




def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_CUBIC)
    return new_target

def resize_output(target, size):
    new_target = np.zeros((target.shape[0], target.shape[1], size, size), np.int32)
    for j in range(target.shape[0]):
        for i, t in enumerate(target[j,:,:,:].data.cpu().numpy()):
            # new_target[j, i, :, :] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
            new_target[j, i, :, :] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_CUBIC)
    return new_target


class FocalLoss(nn.Module):

    # def __init__(self, device, gamma=0, eps=1e-7, size_average=True):
    def __init__(self, gamma=0, eps=1e-7, size_average=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.reduce = reduce
        # self.device = device

    def forward(self, input, target):
        # y = one_hot(target, input.size(1), self.device)
        y = one_hot(target, input.size(1))
        probs = F.softmax(input, dim=1)
        probs = (probs * y).sum(1)  # dimension ???
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.reduce:
            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
        else:
            loss = batch_loss
        return loss


def one_hot(index, classes):
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)


def get_NoGT_target(inputs):
    sfmx_inputs = F.log_softmax(inputs, dim=1)
    target = torch.argmax(sfmx_inputs, dim=1)
    return target

def get_SegLoss_Wt(loss, coef=1):
    return nn.Tanh()(coef * loss)
