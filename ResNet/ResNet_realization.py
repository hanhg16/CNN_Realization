from torch import nn
import torch as t
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,stride,padding=1,bias=False), #padding=x时，相当于原图卷积后的尺寸变为(w+2x)*(h+2x)，此处是为了保持图像的原大小，方便与未卷积的跨层直连单元相加；
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.right = shortcut

    def forward(self,x):
        out = self.left(x)
        residual = x if self.right == None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self,classes_num = 100):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),#经过测试，卷积层是先pad后再进行convalue的，故卷积后单通道feature map的尺寸公式为：h(w) = (high(width) - kernel_size + 2*padding)/stride +1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )

        self.layer1 = self._make_layer(64,64,3)
        self.layer2 = self._make_layer(64,128,4,stride=2)
        self.layer3 = self._make_layer(128,256,6,stride=2)
        self.layer4 = self._make_layer(256,512,3,stride=2)

        self.fc = nn.Linear(512,classes_num)    #全连接层的参数量为:in_channels*out_channels*imghighth*imgwidth，此处之所以能直接=in_channels*out_channels，原因见forward()的return


    def _make_layer(self,in_channes,out_channels,block_num,stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channes,out_channels,1,stride,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        layers = []
        layers.append(ResidualBlock(in_channes,out_channels,stride,shortcut))

        for _ in range(1,block_num):
            layers.append(ResidualBlock(out_channels,out_channels))

        return nn.Sequential(*layers) #列表前加*表示将列表元素解开为独立的变量传入函数

    def forward(self,x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,7)   #用7*7的卷积核将卷积层的输出变为了1*1*512，即图像尺寸为1*1了，用于减少全连接层的参数；
        x = x.view(x.size(0),-1) #将x reshape为1维tensor用于全连接层的计算，其中x.size(0)为batch_size
        return self.fc(x)   #全连接层的参数量为:in_channels*out_channels*imghighth*imgwidth，此处之所以能直接=in_channels*out_channels，是因为前面的平均池化层用7*7的卷积核将卷积层的输出变为了1*1*512，即图像尺寸为1*1了






