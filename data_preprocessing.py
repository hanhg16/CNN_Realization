import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader

show = ToPILImage()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#训练集
trainset = tv.datasets.CIFAR10(root='D:\Projects\CNN_realization\datasets',
                               train=True,
                               transform=transform
                               , download=True)

trainloader = t.utils.data.DataLoader(trainset,
                                      batch_size=4,
                                      shuffle=True,
                                      num_workers=2)

#测试集
