import train_Net as tN
import data_preprocessing as dp
import torch as t
from torch.autograd import Variable


def test_onebatch():
    dataiter = iter(dp.testloader)
    images, labels = dataiter.__next__()
    print('实际的label: ', ' / '.join('%s' % dp.classes[labels[j]] for j in range(4)))
    outputs = tN.net(Variable(images))
    _, predicted = t.max(outputs.data, 1)   #outputs为Variable数据，该数据类型中的data为其数据，除此外还有grad，grad_fn,此处参数1为按行比较，0则为按列比较，因为outputs是个4*10的向量，若按列比较则会输出1*10的向量
    print('预测结果： ', ' / '.join('%s' % dp.classes[predicted[j]] for j in range(4)))


def test_all():
    correct = 0
    total = 0
    for data in dp.testloader:
        images, labels = data

        # Cuda加速
        if t.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = tN.net(Variable(images))
        _, predicted = t.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('测试集中的预测准确率为: %d %'
          '%' %(100*correct/total))


