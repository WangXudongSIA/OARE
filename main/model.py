# Organization: Electrical Engineering and Intelligent Control major, North University of China
# Author: Xudong Wang
# Time：2022/3/7 9:40
# function: The construction of MdehazeNet network

import torch
import torch.nn as nn
import math
from utils import BRelu


class WWTcon(nn.Module):
    def __init__(self):
        super(WWTcon, self).__init__()
        self.RRelu = nn.RReLU()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(3, 9, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(9, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(3, 1, 1, 1, bias=True)

    def forward(self, T):

        T = T.unsqueeze(1)
        # T = torch.cat((T, 1-T), 1)
        # OT = T

        GAM = T
        GAM = (self.conv1(GAM))
        GAM = self.RRelu(GAM)

        GAM = (self.conv2(GAM))
        GAM = self.RRelu(GAM)

        GAM = (self.conv3(GAM))
        GAM = self.RRelu(GAM)

        GAM = (self.conv4(GAM))
        GAM = self.Relu(GAM)

        GAM = 10*GAM + 10

        # GAM = 16

        y = 1 / (1+torch.exp(-GAM*(T-0.5)))
        # y = 1 / (1 + torch.exp(-25 * (T - 0.5)))

        return y


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, 1, 1, dilation=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y

class Aw_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(Aw_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y1 = self.max_pool(x)
        y2 = self.avg_pool(x)
        y1 = self.conv1(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.conv2(y2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = torch.cat((y1, y2), 2)
        y = self.conv3(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        self.RRelu = nn.RReLU()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.BRelu = BRelu()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0,
        #        dilation=1, groups=1,bias=True, padding_mode=‘zeros’)
        #        输入的通道数 输出的通道数 卷积核大小 步长 边界补零
        # 控制卷积核之间的间距 控制输入和输出之间的连接 是否将一个学习到的bias增加输出 字符串类型
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(8, 8, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(16, 8, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(24, 8, 3, 1, 1, bias=True)

        self.conv5 = nn.Conv2d(32, 3, 3, 1, 1, bias=True)

        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1, bias=True)

        self.IN1 = nn.InstanceNorm2d(8, affine=True)
        self.IN2 = nn.InstanceNorm2d(8, affine=True)
        self.IN3 = nn.InstanceNorm2d(8, affine=True)
        self.IN4 = nn.InstanceNorm2d(8, affine=True)

        self.aw1 = Aw_block(32)
        self.aw2 = Aw_block(32)


    def forward(self, x):
        with torch.no_grad():
            xx = x

            x1 = self.RRelu(self.conv1(x))
            x1 = self.IN1(x1)

            x2 = self.RRelu(self.conv2(x1))
            x2 = self.IN2(x2)
            x3 = torch.cat((x1, x2), 1)

            x3 = self.RRelu(self.conv3(x3))
            x3 = self.IN3(x3)
            x4 = torch.cat((x1, x2, x3), 1)

            x4 = self.RRelu(self.conv4(x4))
            x4 = self.IN4(x4)
            Fs = torch.cat((x1, x2, x3, x4), 1)

        GAM = self.aw2(Fs)
        GAM = self.relu(self.conv6(GAM)) + 0.5

        K = self.aw1(Fs)
        K = self.BRelu(self.conv5(K))

        clean_image = K * xx - K + 1

        clean_image = torch.pow(clean_image, GAM, out=None)

        return clean_image
