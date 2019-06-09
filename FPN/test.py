#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os.path as osp

import cv2
import torch
import torch.nn as nn
import yaml
from addict import Dict
from data.data_loading import *
from models.fpn import fpn
# from models.fcn import FCN8
from utils.metric import label_accuracy_hist, hist_to_score


def load_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    network.load_state_dict(torch.load(save_path))
    print("the network load is in " + save_path, file=open("output.txt", "a"))


def save_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    torch.save(network.to("cpu").state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()


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

    # Dataset
    dataset = MultiDataSet(
        CONFIG.ROOT,
        CONFIG.CROPSIZE,
        CONFIG.INSIZE,
        phase="crossvali",
        testFlag=True,
        preload=False
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=False,
    )

    for epoch in range(10, 11, 5):
        # Model
        torch.set_grad_enabled(False)
        model = fpn(CONFIG.N_CLASSES)
        model = nn.DataParallel(model)
        load_network(CONFIG.SAVE_DIR, model, "SateFPN", str(epoch) + "000")
        model.to(device)
        model.eval()

        #visualizer
        # vis = Visualizer(CONFIG.DISPLAYPORT)

        hist = np.zeros((7, 7))
        for i, (data, target) in enumerate(loader):
            print("Process img%d" % i)
            # Image
            data = data.to(device)

            # Propagate forward
            output = model(data)

            # Resize target for {100%, 75%, 50%, Max} outputs
            outImg = cv2.resize(output[0].to("cpu").max(0)[1].numpy(), (target.shape[1],) * 2, interpolation=
                                cv2.INTER_NEAREST)
            if i % 10 == 0:
                save_dir = osp.join(CONFIG.SAVE_DIR, "images")
                cv2.imwrite(osp.join(save_dir, "image"+str(i)+".png"), cv2.cvtColor(inputImgTransBack(data),
                                                                                    cv2.COLOR_RGB2BGR))
                cv2.imwrite(osp.join(save_dir, "predict"+str(i)+".png"), cv2.cvtColor(classToRGB(outImg),
                                                                                      cv2.COLOR_RGB2BGR))
                cv2.imwrite(osp.join(save_dir, "label" + str(i) + ".png"), cv2.cvtColor(classToRGB(target[0].to("cpu")),
                                                                                        cv2.COLOR_RGB2BGR))
            # metric computer
            hist += label_accuracy_hist(target[0].to("cpu").numpy(), outImg, 7)
            # visualizer
            # vis.displayImg(inputImgTransBack(data), classToRGB(output.to("cpu").max(0)[1]),
            #                 classToRGB(target[0].to("cpu")))
        print("hist is {}".format(hist), file=open("output.txt", "a"))
        _, acc_cls, recall_cls, iu, _ = hist_to_score(hist)
        print("accuracy of every class is {}, recall of every class is {}, iu of every class is {}".format(
            acc_cls, recall_cls, iu), file=open("output.txt", "a"))
        print("mean iu is {}".format(np.nansum(iu[1:])/6), file=open("output.txt", "a"))


if __name__ == "__main__":
    main()