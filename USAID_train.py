from utils import *
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from USAID_dataloader import *
from FPN.models.fpn import fpn



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--cropSize", type=int, default=48, help="Image crop size")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--coef_MSE", type=float, default=5e-1, help="Coefficient for MSE in total loss, (1-coef_MSE) for Seg Loss")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--num_of_SegClass", type=int, default=21, help='Number of Segmentation Classes, default VOC = 21')
opt = parser.parse_args()

save_dir = opt.outf
print ('save models to directory : ', save_dir)
if not os.path.exists(save_dir):
    os.mkdirs(save_dir)

def main():
    # Load dataset
    print('Loading dataset ...\n')
    # VOC dataset loading
    dataset_train = MultiDataSet(cropSize=opt.cropSize, testFlag=False, Scale=False)
    dataset_val = MultiDataSet(cropSize=opt.cropSize, testFlag=True, Scale=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=1, batch_size=1, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    print("# of validation samples: %d\n" % int(len(dataset_val)))

    # Denoiser
    net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False).cuda()
    model = nn.DataParallel(net).cuda()

    seg = fpn(opt.num_of_SegClass)
    seg_criterion = FocalLoss(gamma=2).cuda()
    seg = nn.DataParallel(seg).cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,40, 80, 120, 140], gamma=0.1)

    # training
    writer = SummaryWriter(save_dir)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):

        scheduler.step()
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            img_train = data

            model.train()
            seg.train()
            model.zero_grad()
            seg.zero_grad()
            optimizer.zero_grad()

            # training step
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)

            out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)

            # demean segmentation inputs
            seg_input = out_train.data.cpu().numpy()
            for n in range(out_train.size()[0]):
                seg_input[n, :, :, :] = rgb_demean(seg_input[n, :, :, :])
            seg_input = Variable(torch.from_numpy(seg_input).cuda())

            seg_output = seg(seg_input)

            target = (get_NoGT_target(seg_output)).data.cpu()

            target_ = resize_target(target, seg_output.size(2))
            target_ = torch.from_numpy(target_).long()
            target_ = target_.cuda()
            seg_loss = seg_criterion(seg_output, target_)

            for param in seg.parameters():
                param.requires_grad = False

            totalLoss = opt.coef_MSE * loss + (1 - opt.coef_MSE) * seg_loss
            totalLoss.backward()
            optimizer.step()

            if (i+1) % 1000 == 0:
                print("[epoch %d][%d/%d]  [SegClass: %d]  loss: %.4f  PSNR_train: %.4f" %
                    (epoch+1, i+1, len(loader_train), opt.num_of_SegClass, loss.item(), psnr_train))
                # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        ## the end of every 20 epoch， do validation
        if (epoch+1) % 20 == 0:
            model.eval()
            psnr_val = 0
            niqe_val = 0
            ssim_val = 0
            with torch.no_grad():
                for i, data in enumerate(loader_val, 0):

                    img_val = data
                    noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.noiseL / 255.)
                    imgn_val = img_val + noise
                    img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())

                    out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
                    psnr_val += batch_PSNR(out_val, img_val, 1.)
                    ssim_val += batch_SSIM(out_val, img_val, 1.)

                    if epoch == opt.epochs - 1:
                        niqe_val += batch_NIQE(out_val)


                psnr_val /= len(loader_val)
                ssim_val /= len(loader_val)
                writer.add_scalar('PSNR on validation data', psnr_val, epoch)
                torch.save(model.state_dict(), os.path.join(save_dir,
                                                            str(opt.num_of_SegClass)
                                                            + '_USAID_epoch'
                                                            + str(epoch+1) + '_'
                                                            + str(psnr_val) + '.pth'))
                print("\n[epoch %d] [SegClass: %d] PSNR_val: %.2f SSIM_val: %.4f"
                      % (epoch + 1, opt.num_of_SegClass, psnr_val, ssim_val))
                print("**********************************************************************")

                if epoch == opt.epochs - 1:
                    niqe_val /= len(loader_val)

                    torch.save(model.state_dict(),
                               os.path.join(save_dir, str(opt.num_of_SegClass) + '_USAID_final.pth'))
                    print("\n[epoch %d] [SegClass: %d] PSNR_val: %.2f SSIM_val: %.4f NIQE_val: %.4f"
                          % (epoch + 1, opt.num_of_SegClass, psnr_val, ssim_val, niqe_val))
                    print("\n==========  END  ===========")

        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(save_dir, str(opt.num_of_SegClass) + '_USAID_lastest.pth'))



if __name__ == "__main__":
    main()
