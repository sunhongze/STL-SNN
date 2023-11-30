import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from func import *
from params import mnist_para

class MNIST(nn.Module):
    def __init__(self, train=mnist_para.train_thresh, thresh=mnist_para.init_thresh, heterogeneity=mnist_para.hete_thresh, tau=mnist_para.tau, P=10, time_step=mnist_para.time_step):
        super(MNIST, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1   = BatchNorm2d(num_features=128)
        self.lif1  = LIF(train=train, thresh=torch.ones([128,28,28])*thresh, tau=tau, heterogeneity=heterogeneity)
        self.pool1 = AdaptiveMaxPool2d(14)

        self.conv2 = Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2   = BatchNorm2d(num_features=128)
        self.lif2  = LIF(train=train, thresh=torch.ones([128,14,14])*thresh, tau=tau, heterogeneity=heterogeneity)
        self.pool2 = AdaptiveMaxPool2d(7)

        self.drop1 = Dropout(0.5)
        self.fc1 = Linear(7 * 7 * 128, 2048, bias=False)
        self.lif3 = LIF(train=train, thresh=torch.ones([2048])*thresh, tau=tau, heterogeneity=heterogeneity)

        self.drop2 = Dropout(0.5)
        self.fc2 = Linear(2048, 10*P, bias=False)
        self.lif4 = LIF(train=train, thresh=torch.ones([10*P])*thresh, tau=tau, heterogeneity=heterogeneity)

        self.boost = nn.AvgPool1d(P, P)

    def forward(self, input):
        inputs = input.unsqueeze(1).repeat(1,mnist_para.time_step,1,1,1)
        x = inputs.cpu() >= torch.rand(inputs.size())

        x = self.conv1(x.float().cuda())
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.pool2(x)

        x = x.view(x.shape[0], x.shape[1], -1)

        x = self.drop1(x)
        x = self.fc1(x)
        x = self.lif3(x)

        x = self.drop2(x)
        x = self.fc2(x)
        x = self.lif4(x)

        outputs = self.boost(x).mean(1)

        return outputs
