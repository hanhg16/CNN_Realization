import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class dogvscat(Dataset):
    def __init__(self,rootpath,train=True,test=False,transforms=None):
        '''
        get address of all pictures, which contact test,train and verify dataset,
        so you should classify data type  here and then make a list to storage respectively
        '''
        self.test = test
        self.train = train
        # make a list to storage all picture's path,     the os.path.join can joint rootpath and imgnum to a string,      the os.listdir can list all folders or files in the rootpath
        imgs = [os.path.join(rootpath,imgnum) for imgnum in os.listdir(rootpath)]

        #test1: D:/Projects/CNN_realization/dogvscat/data/test1/777.jpg
        if self.test:
            #sorted the list by picture's number , because the list is disordered after os.listdir()
            #the parameter key is used to cpmpare and sort , the split() is used to split string by parameter and [] means which part splited is used,[-1] means last one
            imgs = sorted(imgs,key=lambda x: int(x.split('.')[-2].split('/')[-1]) )

        # train: D:/Projects/CNN_realization/dogvscat/data/train/cat.777.jpg
        else:
            #there are repeating elemrnts,such as cat.66.jpg and dog.66.jpg
            imgs = sorted(imgs,key=lambda x:int(x.split('.')[-2]))

        img_num = len(imgs)

        #divide train into train and verify dataset by 7:3
        if self.test:
            self.imgs = imgs
        elif self.train:
            self.imgs = imgs[:int(0.7*img_num)]
        else:
            self.imgs = imgs[int(0.7*img_num):]

        if transforms is None:

            # compute by traindata_mean_std_compute.py,and the result is storaged at 'dataset_mean_std.txt'
            normalize = T.Normalize(mean=[0.4883,0.4551,0.4170],std=[0.2294,0.2250,0.2252])

            #the transforms of test ,train dataset is different

            #test and verify
            if self.test or not self.train:
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ]) #the function's function refer to pytorch official illustrate
            #train
            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ]) #the function's function refer to pytorch official illustrate

    def __getitem__(self, index):
        '''
        return a picture's data (include tensor and label) by index
        for test data ,return tensor and picture id because of no label,for example,1000.jpg return 1000
        '''
        img_path = self.imgs[index]

        if self.test:
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0

        data = Image.open(img_path)
        data = self.transforms(data)
        return data,label

    def __len__(self):
        '''

        :return: the number of pictures in dataset
        '''
        return len(self.imgs)