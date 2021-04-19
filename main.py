from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
import numpy as np

img = Image.open('nn.Module_practice/imgs/lena.bmp')
to_tensor = ToTensor()
to_pil = ToPILImage()
input = to_tensor(img)

pool = nn.AvgPool2d(2,2)
l = list(pool.parameters())
print(l)

output = pool(V(input))
out = to_pil(output)
out.show()