from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch as t
import torch.nn as nn
import torchvision as tv
from torch.nn import functional as F
from torch.autograd import Variable as V
import numpy as np
import math

from torchvision.models import AlexNet
model = AlexNet()
print(model.state_dict().keys())
for name,parameters in model.named_parameters():
    print(name,end=',')

