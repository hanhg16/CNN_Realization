import data_preprocessing as dp
import train_Net as tN
import torch as t
from torch.autograd import Variable
from test import test_all as test


def train(epoch=3):
    for epo in range(epoch):

        running_loss = 0.0
        for i, data in enumerate(dp.trainloader, 0):  # 从下标0作为初始索引开始，遍历元素,由于设置的batch_size = 4，因此每一个i和data都是包含4张图片的集合

            # 输入数据
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            '''
            #测验证是不是包含4张图片的数
            print(labels)
            print('--------end-----')
            '''
            # 梯度清零
            tN.optimizer.zero_grad()

            # Cuda加速
            if t.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward + backward
            outputs = tN.net(inputs)
            loss = tN.criterion(outputs, labels)
            loss.backward()

            # 更新参数
            tN.optimizer.step()

            # 打印日志信息
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %d] loss: %f '
                      % (epo + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Training Finished')


if __name__ == '__main__':
    train(1)




'''
CPU训练时间：3min 16sec;epoch==5
GPU训练时间：3min 09sec;epoch==5
可见此时加速效果并不明显
'''