from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch as t
import torch.nn as nn
from torch.autograd import Variable as V

to_tensor = ToTensor() #因为ToTensor是个class，所以只能先生成对象，再进行操作
to_pil = ToPILImage()
lena = Image.open('imgs/lena.bmp')

input = to_tensor(lena).unsqueeze(0)  # 从3维变到4维，因为第一维是batch_size，此处加上第一维表示batch_size为1

kernel = t.ones(3, 3)/9
#kernel[1][1] = -1
print(kernel)
conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)

for name, parameters in conv.named_parameters():
    print(name, ':', parameters.size())

output = conv(V(input))
out = to_pil(output.data.squeeze(0))
out.show()


# 将卷积后的图片根据kernel_size和kernel_data命名并保存
import numpy as np

#因为pytorch里的tensor.size()的数据类型为专用的，转换为字符串输出很麻烦，因此先转换为numpy数组，numpy数值里的array.shape对应tensor.size()，且数据类型为tuple，方便for循环读出
kernel_np = np.array(kernel)

#由于保存路径为字符串，因此此处生成先文件名与路径的字符串；
#python中不同字符串可以通过'+'拼接，通过''.join()将循环数据按''中的分隔符拼接，注意join()括号里的字内容要先转换为字符串，方法如下所示；另外，其中的循环允许嵌套，方法如'''...'''中所示
'''
s = ','.join('%s' % j for i in kernel_np for j in i)
print(s)
'''
path = 'imgs/lena_kernelsize' + '(' + ','.join('%s' % i for i in kernel_np.shape) + ')_' + 'kerneldata'
# 用for循环拼接字符串，达到[(1),(2),(3)][...][...]的效果，方便分辨kernel的行列
for c in range(kernel_np.shape[0]):
    path = path + '[' + ','.join('(%s)' %j  for j in kernel_np[c]) + ']'

path += '.png'
#将之前生成的字符串作为存储路径保存
out.save(path)