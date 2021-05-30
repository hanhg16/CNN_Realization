import os
from PIL import Image
from models import BasicModule

rootpath = 'D:/Projects/CNN_realization/dogvscat/data/test1/'
imgs = [os.path.join(rootpath,imgnum) for imgnum in os.listdir(rootpath)]
x=[]
for i in imgs:
    x.append(i.split('.')[-2].split('/')[-1])

x= sorted(x,key=lambda s:int(s))
print(x)