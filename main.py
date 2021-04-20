from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
import numpy as np
import math

module = nn.Module()
print(getattr(module,'training'))
#module.__getattr__('training')
module.param = t.zeros(2,2)
module.param1 = nn.Parameter(t.ones(3,3))

print(getattr(module,'param'))
print(getattr(module,'param1'))
print(module.__getattr__('param1'))

t.save(module.state_dict(),'save_module/net.pth')