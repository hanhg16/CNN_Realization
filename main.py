from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch as t
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable as V
import numpy as np
import math

'''
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,stride,padding=1,bias=False), #padding=x时，相当于原图卷积后的尺寸变为(w+2x)*(h+2x)，此处是为了保持图像的原大小，方便与未卷积的跨层直连单元相加；
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.right = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,1,stride,bias=False),
        nn.BatchNorm2d(out_channels)
    )

    def forward(self,x):
        out = self.left(x)
        residual = x if self.right == None else self.right(x)
        out += residual
        return F.relu(out)


net = ResidualBlock(1,3)

x = t.randn(1,1,50,50)
y = net(x)
print(y.size())
'''
fc = nn.Linear(3*2*2,4)
param = list(fc.parameters())
print(param[0])
