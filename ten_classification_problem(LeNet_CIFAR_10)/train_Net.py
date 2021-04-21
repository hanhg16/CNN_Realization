import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


#定义训练网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1) #将x reshape为1维tensor以用于之后的全连接层计算，此处的x.size(0)是x的batch_size，即reshape前x的size为[4,16,5,5],view后为[4,400]
        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
        return x

net = Net()
# Cuda加速
if t.cuda.is_available():
    net.cuda()

#定义损失函数和优化器
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)