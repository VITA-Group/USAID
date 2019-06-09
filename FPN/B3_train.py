#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os.path as osp
import os
import time
# import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
# from tqdm import tqdm
# from utils.visualizer import Visualizer
# from data.data_loading import *
from data.B3_dataloader import *
from models.fpn import fpn
from utils.loss import CrossEntropyLoss2d, SoftCrossEntropyLoss2d, FocalLoss
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def save_image(numpy, savepath):
    Out_ = (np.transpose(numpy, (1, 2, 0)) * 255.0)
    Out_ = Out_[..., ::-1]
    cv2.imwrite(savepath, Out_)

def load_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    # network.load_state_dict(torch.load(save_path))
    network.load_state_dict(torch.load('/hdd2/sharonwang/DnCNN/FPN/checkpoints/B3_noGT/final_net_SateFPN.pth'))


def save_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    # torch.save(network.to("cpu").state_dict(), save_path)
    torch.save(network.state_dict(), save_path)
    if not torch.cuda.is_available():
        network.to('cpu')


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return new_target

def get_NoGT_target(inputs):
    sfmx_inputs = F.log_softmax(inputs, dim=1)
    target = torch.argmax(sfmx_inputs, dim=1)
    return target


def main():

    config = "config/cocostuff.yaml"
    cuda = True
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(config)))
    CONFIG.SAVE_DIR = osp.join(CONFIG.SAVE_DIR, CONFIG.EXPERIENT)
    CONFIG.LOGNAME = osp.join(CONFIG.SAVE_DIR, "log.txt")
    if not os.path.exists(CONFIG.SAVE_DIR):
        os.mkdir(CONFIG.SAVE_DIR)

    # Dataset
    dataset = MultiDataSet(cropSize=50, inSize = 500, testFlag=False, preload=True)
    print("# of training samples: %d\n" % int(len(dataset)))
    # dataset = MultiDataSet(
    #     CONFIG.ROOT,
    #     CONFIG.CROPSIZE,
    #     CONFIG.INSIZE,
    #     preload=False
    # )

    # DataLoader
    if CONFIG.RESAMPLEFLAG:
        batchSizeResample = CONFIG.BATCH_SIZE
        CONFIG.BATCH_SIZE = 1

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Model
    model = fpn(CONFIG.N_CLASSES)
    model = nn.DataParallel(model)

    # read old version
    if CONFIG.ITER_START != 1:
        load_network(CONFIG.SAVE_DIR, model, "SateFPN", "latest")
        print("load previous model succeed, training start from iteration {}".format(CONFIG.ITER_START))
    model.to(device)

    # Optimizer
    optimizer = {
        "sgd": torch.optim.SGD(
            # cf lr_mult and decay_mult in train.prototxt
            params=[
                {
                    "params": model.parameters(),
                    "lr": CONFIG.LR,
                    "weight_decay": CONFIG.WEIGHT_DECAY,
                }
            ],
            momentum=CONFIG.MOMENTUM,
        )
    }.get(CONFIG.OPTIMIZER)

    # Loss definition
    # criterion = FocalLoss(device, gamma=2)
    criterion = FocalLoss(gamma=2)
    criterion.to(device)

    #visualizer
    # vis = Visualizer(CONFIG.DISPLAYPORT)

    model.train()
    iter_start_time = time.time()
    save_loss = []
    counter = 0
    save_dir = '/ssd1/sharonwang/0916-trainSegMap'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for iteration in range(CONFIG.ITER_START, CONFIG.ITER_MAX + 1):
        # Set a learning rate
        poly_lr_scheduler(
            optimizer=optimizer,
            init_lr=CONFIG.LR,
            iter=iteration - 1,
            lr_decay_iter=CONFIG.LR_DECAY,
            max_iter=CONFIG.ITER_MAX,
            power=CONFIG.POLY_POWER,
        )

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        iter_loss = 0

        for i in range(1, CONFIG.ITER_SIZE + 1):
            if not CONFIG.RESAMPLEFLAG:
                try:
                    data = next(loader_iter)
                    # data, target = next(loader_iter)
                except:
                    loader_iter = iter(loader)
                    data = next(loader_iter)
                    # data, target = next(loader_iter)
            else:
                cntFrame = 0
                # clDataStart = time.time()
                clCnt = 0
                while cntFrame < batchSizeResample:
                    clCnt += 1
                    try:
                        dataOne = next(loader_iter)
                        # dataOne, targetOne = next(loader_iter)
                    except:
                        loader_iter = iter(loader)
                        dataOne = next(loader_iter)
                        # dataOne, targetOne = next(loader_iter)

                    hist = np.bincount(dataOne.numpy().flatten(), minlength=21)
                    hist = hist / np.sum(hist)
                    if np.nanmax(hist) <= 0.70:
                        if cntFrame == 0:
                            data = dataOne
                            # target = targetOne
                        else:
                            data = torch.cat([data, dataOne])
                            # target = torch.cat([target, targetOne])
                        cntFrame += 1
                # print("collate data takes %.2f sec, collect %d time" % (time.time() - clDataStart, clCnt))

            # Image
            data = data.to(device)
            # Propagate forward
            output = model(data)
            # Loss
            loss = 0
            target = get_NoGT_target(output).data.cpu()

            ##################################################
            seg_rgb = classToRGB(target.permute((1, 2, 0)))
            seg_rgb = cv2.resize(seg_rgb, (seg_rgb.shape[0] * 4, seg_rgb.shape[1] * 4))
            seg_rgb = torch.from_numpy(seg_rgb).cuda().float() / 255.
            seg_rgb = torch.unsqueeze(seg_rgb.permute((2, 0, 1)), 0)
            # print(target_seg_map.shape)
            # print(seg_rgb.shape)
            # print(imgn_train.shape)
            img_tensor = torch.cat((data, seg_rgb), 3)


            save_image(img_tensor[0].cpu().float().detach().numpy(),
                       os.path.join(save_dir, str(counter) + '.png'))
            counter += 1
            ##################################################
            # Resize target for {100%, 75%, 50%, Max} outputs
            target_ = resize_target(target, output.size(2))
            # classmap = class_to_target(target_, CONFIG.N_CLASSES)
            # target_ = label_bluring(classmap)  # soft crossEntropy target
            target_ = torch.from_numpy(target_).long()
            target_ = target_.to(device)

            for p in model.module.resnet.parameters():
                p.requires_grad = False
            # Compute crossentropy loss
            loss += criterion(output, target_)
            # Backpropagate (just compute gradients wrt the loss)
            loss /= float(CONFIG.ITER_SIZE)
            loss.backward()

            iter_loss += float(loss)
        save_loss.append(iter_loss)

        # Update weights with accumulated gradients
        optimizer.step()
        # Visualizer and Summery Writer
        if iteration % CONFIG.ITER_TF == 0:
            print("itr {}, loss is {}".format(iteration, iter_loss))
            # print("itr {}, loss is {}".format(iteration, iter_loss), file=open(CONFIG.LOGNAME, "a"))  #
            # print("time taken for each iter is %.3f" % ((time.time() - iter_start_time)/iteration))
            # vis.drawLine(torch.FloatTensor([iteration]), torch.FloatTensor([iter_loss]))
            # vis.displayImg(inputImgTransBack(data), classToRGB(outputs[3][0].to("cpu").max(0)[1]),
            #                classToRGB(target[0].to("cpu")))
        # Save a model
        if iteration % CONFIG.ITER_SNAP == 0:
            save_network(CONFIG.SAVE_DIR, model, "SateFPN", iteration)

        # Save a model
        if iteration % 100 == 0:
            save_network(CONFIG.SAVE_DIR, model, "SateFPN", "latest")

    save_network(CONFIG.SAVE_DIR, model, "SateFPN", "final")
    save_loss_ary = np.asarray(save_loss)
    np.save(CONFIG.SAVE_DIR + '/save_loss.npy', save_loss_ary)


if __name__ == "__main__":
    main()
