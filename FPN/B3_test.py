import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# from models import DnCNN
from utils import *
from VOC_dataloader import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from FPN.models.fpn import fpn
from FPN.utils.metric import label_accuracy_hist, hist_to_score
from data.B3_dataloader import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/BL3_Diff_SegClass", help='path of log files')
parser.add_argument("--saved_model", type=str, default="21_VOC_BL3_Denoiser", help='name of saved model')
# parser.add_argument("--test_data", type=str, default='CBSD68', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=35, help='noise level used on test set')
opt = parser.parse_args()
# print('sigma = ', opt.test_noiseL, 'Used baseline:', opt.saved_model)
#
# save_dir = 'segmentation_saved_imgs/' + opt.saved_model + '_' + str(opt.test_noiseL)
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)


def main():
    torch.cuda.manual_seed(1234)
    print('Loading model ...\n')
    # net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    # model = nn.DataParallel(net).cuda()
    # model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.saved_model+'.pth')))
    # model.eval()

    seg = fpn(21)
    seg = nn.DataParallel(seg).cuda()
    seg.eval()
    seg.load_state_dict(torch.load('/hdd2/sharonwang/DnCNN/FPN/checkpoints/B3_noGT/final_net_SateFPN.pth'))

    dataset_val = MultiDataSet(cropSize=50, inSize=500, testFlag=True, preload=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print("# of validation samples: %d\n" % int(len(dataset_val)))

    # psnr_val = 0
    # ssim_val = 0
    # niqe_val = 0
    mean_iu_val = 0
    hist = np.zeros((21, 21))
    for i, data in enumerate(loader_val, 0):
        '''
        img_val = data[0]
        target = data[1]
        noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        # if opt.mode == 'S':
        #     noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.noiseL / 255.)
        # if opt.mode == 'B':
        #     noise = torch.zeros(img_val.size())
        #     stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
        #     for n in range(noise.size()[0]):
        #         sizeN = noise[0, :, :, :].size()
        #         noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
        imgn_val = img_val + noise
        img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())

        out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)

        psnr = batch_PSNR(out_val, img_val, 1.)
        ssim = batch_SSIM(out_val, img_val, 1.)
        niqe = batch_NIQE(out_val)

        psnr_val += psnr
        ssim_val += ssim
        niqe_val += niqe

        img_2_save = transforms.ToPILImage()(img_val[0].data.cpu())
        img_2_save.save(os.path.join(save_dir, "img_" + str(i) + '.png'))

        denoised_img = transforms.ToPILImage()(out_val[0].data.cpu())
        denoised_img.save(os.path.join(save_dir, "denoised_"+ str(i) + "_" + str(niqe) +'.png'))
        '''

        # seg_input = out_val.data.cpu().numpy()
        seg_input = data.data.cpu().numpy()
        for n in range(seg_input.shape[0]):
            seg_input[n, :, :, :] = rgb_demean(seg_input[n, :, :, :])
        seg_input = Variable(torch.from_numpy(seg_input).cuda())
        output = seg(seg_input)

        # Resize target for {100%, 75%, 50%, Max} outputs
        outImg = cv2.resize(output[0].to("cpu").max(0)[1].numpy(), (500,500), interpolation=
        cv2.INTER_NEAREST)

        # metric computer
        # img_hist = label_accuracy_hist(target[0].to("cpu").numpy(), outImg, 21)
        # mean_iu = np.nanmean(np.diag(img_hist) / (img_hist.sum(axis=1) + img_hist.sum(axis=0) - np.diag(img_hist)))
        # hist += img_hist
        # mean_iu_val += mean_iu

        seg_map = cv2.cvtColor(classToRGB(outImg),cv2.COLOR_RGB2BGR)
        # img_tensor = torch.cat((imgn_train, img_train, denoised_gradient, seg_gradient, seg_rgb), 3)


        # cv2.imwrite(os.path.join(save_dir, "image" + str(i) + ".png"), cv2.cvtColor(inputImgTransBack(img_val),
        #                                                                         cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join('/ssd1/sharonwang/0916', "predict_" + str(i) + ".png"), seg_map)
        # cv2.imwrite(os.path.join(save_dir, "label_" + str(i) + ".png"), cv2.cvtColor(classToRGB(target[0].to("cpu")),
        #                                                                         cv2.COLOR_RGB2BGR))

        # print("[Image %d]  PSNR: %.4f  SSIM: %.4f  NIQE: %.4f  IOU: %.4f" % (i , psnr, ssim, niqe, mean_iu))


    # psnr_val /= len(loader_val)
    # ssim_val /= len(loader_val)
    # niqe_val /= len(loader_val)
    # mean_iu_val /= len(loader_val)



    # print("\nPSNR on test data %f" % psnr_val)
    # print("\nSSIM on test data %f" % ssim_val)
    # print("\nNIQE on test data %f" % niqe_val)
    # print("\nMean IU on test data %f" % mean_iu_val)
    # print('sigma = ', opt.test_noiseL, 'Used baseline:', opt.saved_model)
    print("***********************************************************")
    #
    # output_file_name = opt.saved_model + '_output_' + str(opt.test_noiseL) + '.txt'
    #
    # print("hist is {}".format(hist), file=open(output_file_name, "a"))
    # _, acc_cls, recall_cls, iu, _ = hist_to_score(hist)
    # print("accuracy of every class is {}, recall of every class is {}, iu of every class is {}".format(
    #     acc_cls, recall_cls, iu), file=open(output_file_name, "a"))
    # print("mean iu is {}".format(np.nansum(iu[1:]) / 21), file=open(output_file_name, "a"))



if __name__ == "__main__":
    main()
