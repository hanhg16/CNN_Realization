import torch as t
import time

'''
基础模型，封装了nn.Module,主要提供save和load两个函数，后续模型通过继承改模型来进行save和load
'''

class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self)).split('.')[-1].split('\'')[-2]
       # self.conv1 = t.nn.Conv2d(3,6,5)

    def load(self,path):
        self.load_state_dict(t.load(path))

    def save(self,name = None):
        '''
        保存模型，使用“模型名字+时间“作为文件名
        如AlexNet_0530_16:46:30.pth
        '''
        if name is None:

            prefix = 'D:/Projects/CNN_realization/dogvscat/checkpoints/' + self.model_name +'_'
            name = time.strftime(prefix + '%m%d_%H：%M：%S.pth')
            t.save(self.state_dict(),name)
            return name
