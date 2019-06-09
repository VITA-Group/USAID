import torch.nn as nn
import torch.nn.functional as F
import torch


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


# def one_hot(index, classes, device):
def one_hot(index, classes):
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    # def __init__(self, device, gamma=0, eps=1e-7, size_average=True):
    def __init__(self, gamma=0, eps=1e-7, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        # self.device = device

    def forward(self, input, target):
        # y = one_hot(target, input.size(1), self.device)
        y = one_hot(target, input.size(1))
        probs = F.softmax(input, dim=1)
        probs = (probs * y).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class SoftCrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss2d, self).__init__()

    def forward(self, inputs, targets):
        loss = 0
        inputs = -F.log_softmax(inputs, dim=1)
        for index in range(inputs.size()[0]):
            loss += F.conv2d(inputs[range(index, index+1)], targets[range(index, index+1)])/(targets.size()[2] *
                                                                                             targets.size()[3])
        return loss


