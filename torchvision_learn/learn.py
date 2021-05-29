from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid,save_image
from torch import nn
from torch.utils.data import DataLoader
import torch as t
from PIL import Image

to_img = transforms.ToPILImage()
transform = transforms.ToTensor()

dataset = datasets.MNIST('data/',download=True,train=False,transform=transform)
dataloader = DataLoader(dataset,shuffle=True,batch_size=16)
dataiter = iter(dataloader)
img = make_grid(next(dataiter)[0],4)

save_image(img,'data/a.png')
img = Image.open('data/a.png')
img.show()