import torch
import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import json
import random
from collections import OrderedDict

class TrainDataset(Dataset):
    def __init__(self, dataPath, transformImage=None):
        """ Intialize the dataset >.<"""
        self.dataPath = dataPath
        self.transformImage = transformImage
        self.videos = sorted(os.listdir(self.dataPath))
        self.query = 'cast.json'
        self.cand = 'candidate.json'
        self.queryDir = 'cast'
        self.candDir = 'candidates'
        self.datas = []
        self.labels = []
        self.iden = {}
        total = 0
        for index in range(len(self.videos)):
            with open(os.path.join(self.dataPath, self.videos[index], self.query)) as f:
                query = OrderedDict(sorted(json.load(f).items(), key=lambda d:d[0]))
            with open(os.path.join(self.dataPath, self.videos[index], self.cand)) as f:
                cand = OrderedDict(sorted(json.load(f).items(), key=lambda d:d[0]))
            for qk,qv in query.items():
                if qv not in self.iden:
                    self.iden[qv] = total
                    total += 1
                self.datas.append(qk)
                self.labels.append(self.iden[qv])
                for ck,cv in cand.items():
                    if qv == cv:
                        self.datas.append(ck)
                        self.labels.append(self.iden[qv])
                #total += 1
        self.len = len(self.datas)
    # I did not check if lower than sampleSize 
    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]

        if self.transformImage is not None:
            img = Image.open(os.path.join(os.path.dirname(self.dataPath), data))
            img = img.convert('RGB')
            img = self.transformImage(img)

        return img, torch.tensor(label)

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class ValidDataset(Dataset):
    def __init__(self, dataPath, transformImage=None):
        """ Intialize the dataset >.<"""
        self.dataPath = dataPath
        self.transformImage = transformImage
        self.videos = sorted(os.listdir(self.dataPath))
        self.queryDir = 'cast'
        self.candDir = 'candidates'
        self.len = len(self.videos)

    # allow 1 batch size only
    def __getitem__(self, index):
        queryImg = []
        candImg = []
        queryTensor = []
        candTensor = []
        labels = []

        queryImg = sorted(glob.glob(os.path.join(self.dataPath, self.videos[index], self.queryDir, '*.jpg')))
        candImg = sorted(glob.glob(os.path.join(self.dataPath, self.videos[index], self.candDir, '*.jpg')))

        if self.transformImage is not None:
            for fn in queryImg:
                img = Image.open(fn)
                img = img.convert('RGB')
                img = self.transformImage(img)
                queryTensor.append(img)
            for fn in candImg:
                img = Image.open(fn)
                img = img.convert('RGB')
                img = self.transformImage(img)
                candTensor.append(img)
        return torch.stack(queryTensor, 0), torch.stack(candTensor, 0), queryImg, candImg

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class SubqueryDataset(Dataset):
    def __init__(self, subquery, queryNum, dataPath, transformImage=None):
        """ Intialize the dataset >.<"""
        self.dataPath = dataPath
        print(self.dataPath)
        self.transformImage = transformImage
        self.videos = sorted(os.listdir(self.dataPath))
        self.queryDir = 'cast'
        self.candDir = 'candidates'
        self.len = len(self.videos)
        self.subquery = subquery
        self.queryNum = queryNum

    # allow 1 batch size only
    def __getitem__(self, index):
        queryImg = []
        candImg = []
        queryTensor = []
        candTensor = []
        labels = []
        num = []

        # queryImg = sorted(glob.glob(os.path.join(self.dataPath, self.videos[index], self.queryDir, '*.jpg')))
        queryImg = []
        with open(os.path.join(self.subquery,os.path.basename(self.dataPath.strip('/')),self.videos[index]+'.txt')) as f:
            for line in f:
                orig_path=line.strip()
                f_name = os.path.basename(orig_path)
                f_path = os.path.join(self.dataPath,self.videos[index],self.candDir,f_name) 
                queryImg.append(f_path)
        with open(os.path.join(self.subquery,os.path.basename(self.dataPath.strip('/')),self.videos[index]+'_valid.txt')) as f:
            for line in f:
                num.append(line.strip())

        candImg = sorted(glob.glob(os.path.join(self.dataPath, self.videos[index], self.candDir, '*.jpg')))
        

        if self.transformImage is not None:
            for fn in queryImg:
                img = Image.open(fn)
                img = img.convert('RGB')
                img = self.transformImage(img)
                queryTensor.append(img)
            for fn in candImg:
                img = Image.open(fn)
                img = img.convert('RGB')
                img = self.transformImage(img)
                candTensor.append(img)
        queryImg = sorted(glob.glob(os.path.join(self.dataPath, self.videos[index], self.queryDir, '*.jpg')))
        return torch.stack(queryTensor, 0), torch.stack(candTensor, 0), queryImg, candImg, num

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class MultiLossDataset(Dataset):
    def __init__(self, dataPath, transformImage=None):
        """ Intialize the dataset >.<"""
        self.dataPath = dataPath
        self.transformImage = transformImage
        self.videos = sorted(os.listdir(self.dataPath))
        self.query = 'cast.json'
        self.cand = 'candidate.json'
        self.queryDir = 'cast'
        self.candDir = 'candidates'
        self.datas = []
        self.domains = []
        self.labels = []
        self.iden = {}
        total = 0
        for index in range(len(self.videos)):
            with open(os.path.join(self.dataPath, self.videos[index], self.query)) as f:
                query = OrderedDict(sorted(json.load(f).items(), key=lambda d:d[0]))
            with open(os.path.join(self.dataPath, self.videos[index], self.cand)) as f:
                cand = OrderedDict(sorted(json.load(f).items(), key=lambda d:d[0]))
            for qk,qv in query.items():
                if qv not in self.iden:
                    self.iden[qv] = total
                    total += 1
                for ck,cv in cand.items():
                    if qv == cv:
                        self.domains.append(qk)
                        self.datas.append(ck)
                        self.labels.append(self.iden[qv])
                #total += 1
        self.len = len(self.datas)
    # I did not check if lower than sampleSize 
    def __getitem__(self, index):
        data = self.datas[index]
        domain = self.datas[index]
        label = self.labels[index]

        if self.transformImage is not None:
            img = Image.open(os.path.join(os.path.dirname(self.dataPath), data))
            img = img.convert('RGB')
            img = self.transformImage(img)

            d_img = Image.open(os.path.join(os.path.dirname(self.dataPath), domain))
            d_img = d_img.convert('RGB')
            d_img = self.transformImage(d_img)
        return img, d_img, torch.tensor(label)

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
class TripletTrainDataset(Dataset):
    def __init__(self, dataPath, transformImage=None, mix=False):
        """ Intialize the dataset >.<"""
        if not mix:
            self.dataPath = dataPath[0]
            self.videos = sorted(glob.glob(os.path.join(self.dataPath,'*')))
        else:
            self.dataPath = dataPath[0]
            self.videos = sorted(glob.glob(os.path.join(dataPath[0],'*'))+glob.glob(os.path.join(dataPath[1],'*')))
        self.transformImage = transformImage
        self.query = 'cast.json'
        self.cand = 'candidate.json'
        self.queryDir = 'cast'
        self.candDir = 'candidates'
        self.datas = []
        self.labels = []
        self.iden = {}
        total = 0
        for index in range(len(self.videos)):
            with open(os.path.join(self.videos[index], self.query)) as f:
                query = OrderedDict(sorted(json.load(f).items(), key=lambda d:d[1]))
            with open(os.path.join(self.videos[index], self.cand)) as f:
                cand = OrderedDict(sorted(json.load(f).items(), key=lambda d:d[1]))
            for qk,qv in query.items():
                if qv not in self.iden:
                    self.iden[qv] = total
                    total += 1
                # self.datas.append(qk)
                # self.labels.append(self.iden[qv])
                data = []
                label = []
                data.append(qk)
                label.append(self.iden[qv])
                for ck,cv in cand.items():
                    if qv == cv:
                        data.append(ck)
                        label.append(self.iden[qv])

                datas = [data[i:i+4] for i in range(0,len(data),4)]
                labels = [label[i:i+4] for i in range(0,len(label),4)]
                for i in range(len(datas)):
                    if len(datas[i])==4:
                        self.datas.append(datas[i])
                        self.labels.append(labels[i])
                # print(self.labels,'\n')
                #total += 1
        self.classnum=total
        self.len = len(self.datas)
    # I did not check if lower than sampleSize 
    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]
        # print(label,'one item')
        images=[]

        if self.transformImage is not None:
            for i in range(len(data)):
                img = Image.open(os.path.join(os.path.dirname(self.dataPath), data[i]))
                img = img.convert('RGB')
                img = self.transformImage(img)
                images.append(img)
        images = torch.stack(images,0)
        label = torch.tensor(label)

        return images, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
    def get_classnum(self):
        return self.classnum
        


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    train_pipeline = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    trainset = TrainDataset('/home/r07921059/DLCV/final-sitlattepapa/final_data/train', train_pipeline)
    trainset_loader = DataLoader(trainset, batch_size=8, shuffle=False, num_workers=1)
    for (img, lb) in trainset_loader:
        print(img.shape)
        print(lb.shape)
        break
