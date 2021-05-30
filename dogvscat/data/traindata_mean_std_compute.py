from PIL import Image
import os
import torchvision.transforms as T
import numpy as np
import torch as t

to_tensor = T.ToTensor()
to_img = T.ToPILImage()

storage_path = 'D:/Projects/CNN_realization/dogvscat/data/dataset_mean_std.txt'
rootpath = 'D:/Projects/CNN_realization/dogvscat/data/train/'
imgs = [os.path.join(rootpath,img) for img in os.listdir(rootpath)]
img_num = len(imgs)


mean = [0,0,0]
std = [0,0,0]


for index in imgs:
    img = Image.open(index)
    img = to_tensor(img)
    for i in range(3):
        mean[i] += img[i,:,:].mean()
        std[i] += img[i,:,:].std()

mean = [x/img_num for x in mean]
std = [x/img_num for x in std]

with open(storage_path,'w') as file:
    file.write('mean = '+str(mean)+'\n')
    file.write('std = '+str(std)+'\n')