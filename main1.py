from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch as t
import torch.nn as nn
from torch.autograd import Variable as V

to_tensor = ToTensor() #因为ToTensor是个class，所以只能先生成对象，再进行操作
to_pil = ToPILImage()
cat = Image.open('nn.Module_practice/imgs/cat.jpg')

input = to_tensor(cat)

output = t.zeros(1,input.size(1),input.size(2))
for i in range(input.size(1)):
    for j in range(input.size(2)):
        output[0][i][j] = (input[0][i][j] + input[1][i][j] + input[2][i][j])/3

print(output)

input.resize_(1,input.size(1),input.size(2))
print(input)

input = to_pil(input)
output = to_pil(output)
input.show()
output.show()

input.save('nn.Module_practice/imgs/cat_resize.png')
input.save('nn.Module_practice/imgs/cat_mean.png')







