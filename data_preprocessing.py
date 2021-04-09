import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

show = ToPILImage()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 训练集
trainset = tv.datasets.CIFAR10(root='D:\Projects\CNN_realization\datasets',
                               train=True,
                               download=True,
                               transform=transform)

trainloader = t.utils.data.DataLoader(trainset,
                                      batch_size=4,
                                      shuffle=True,
                                      num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(root='D:\Projects\CNN_realization\datasets',
                              train=False,
                              download=True,
                              transform=transform)

testloader = t.utils.data.DataLoader(testset,
                                     batch_size=4,
                                     shuffle=False,
                                     num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


'''
#从内存里输出图片的测试，第一张为数据集里的照片，第二张为随机生成向量画出来的照片，第三张为第二张reshape的结果
(data,label) = trainset[10]
c=show((data+1)/2).resize((100,100))
print(data)
print(c)
plt.imshow(c)
plt.show()

x = t.rand(3,32,32) #在0，1之间随机初始化3*32*32的向量
print(x)
x1 = show(x)    #将向量转换为PIL Image格式
print(x1)
plt.imshow(x1)  #利用matplotlib包对图片进行绘制，绘制成功后，返回一个matplotlib类型的数据
plt.show()  #显示图片

x1 = show(x).resize((100,100))
plt.imshow(x1)
plt.show()
'''



