import argparse
import os
import numpy as np
import math
import sys
import csv
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import *
from tqdm import tqdm
from model import *
from triplet_loss import *
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=25, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--train_data", type=str, default='final_data/train', help="videoPath")
parser.add_argument("--valid_data", type=str, default='final_data/val', help="videoPath")
parser.add_argument("--mode", type=str, default='train', help="mode", choices=['train','test'])
parser.add_argument("--feature_batch", type=int, default=64, help="size of the batches")
parser.add_argument("--model", type=str, default='./final_model/REID_trip_336_192_tmixv.pkl', help="model path")
parser.add_argument("--ntriplet", type=int, default=4, help="nums of tripletSample of one cast")
parser.add_argument("--parallel", type=bool, default=False, help="use multi-gpu or not")
parser.add_argument("--mix_training", type=bool, default=False, help="train with both training and validation data")
opt = parser.parse_args()
print(opt)

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

opt.train_data=opt.train_data.strip('/')
opt.valid_data=opt.valid_data.strip('/')

mix_datapath=[opt.train_data,opt.valid_data]
trainDataset = TripletTrainDataset(
                dataPath=mix_datapath,
                transformImage=transforms.Compose([transforms.Resize((336,192), interpolation=3),
                        transforms.RandomHorizontalFlip(p=0.5), 
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                mix=opt.mix_training 
                        )
num_class=trainDataset.get_classnum()

trainDataloader = torch.utils.data.DataLoader(
    trainDataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

validDataset = ValidDataset(
                dataPath=opt.valid_data,
                transformImage=transforms.Compose([transforms.Resize((336,192), interpolation=3), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

validDataloader = torch.utils.data.DataLoader(
    validDataset,
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# Loss function

# criterion = nn.TripletMarginLoss(margin=1.0, p=2).to(device)
criterion = TripletLoss(margin=1.0).to(device)
CE = nn.CrossEntropyLoss().to(device)


# Initialize model && to gpu
if opt.parallel:
    feature_extractor = nn.DataParallel(ft_net(num_class)).to(device)
else: 
    feature_extractor = ft_net(num_class).to(device)

# ct = 0
# for child in feature_extractor.model.children():
#     ct += 1
#     if ct < 7:
#         for param in child.parameters():
#             param.requires_grad = False
# Optimizers
optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=opt.lr, weight_decay=5e-4)

for epoch in range(1, opt.n_epochs+1):
    feature_extractor.train()
    trange = tqdm(enumerate(trainDataloader), total=len(trainDataloader), desc='Train')
    train_loss = 0
    # each batch contain one person, each person caontain ntriplet
    # for i, (anchor, positive, negative) in trange:
    for i, (images, labels) in trange:
        # anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        # optimizer.zero_grad()
        # feature_a , _ = feature_extractor(anchor.view(-1, 3, 256, 128))
        # feature_p , _ = feature_extractor(positive.view(-1, 3, 256, 128))
        # feature_n , _ = feature_extractor(negative.view(-1, 3, 256, 128))
        # loss = criterion(feature_a, feature_p, feature_n)
        images, labels = images.to(device), labels.to(device)
        # print(images.size(), labels.size())
        optimizer.zero_grad()
        embedding,outputs = feature_extractor(images.view(-1, 3, 336, 192))
        embeddingnorm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        embedding = embedding.div(embeddingnorm.expand_as(embedding))   
        loss = criterion(embedding, labels.view(embedding.size(0),-1))
        # print(outputs.size(), labels.size())
        loss += CE(outputs, labels.view(embedding.size(0)))
        train_loss += loss
        loss.backward()
        optimizer.step()
        trange.set_postfix(epoch=epoch, loss=train_loss.item()/(i+1))
    os.makedirs(os.path.dirname(opt.model), exist_ok=True)
    
    if opt.parallel:
        save_feature_dict=OrderedDict()
        
        for k, v in feature_extractor.state_dict().items():
            name = k[7:] # remove `module.`
            save_feature_dict[name] = v
    else:
        save_feature_dict=feature_extractor.state_dict()
        
    torch.save(save_feature_dict, opt.model+str(epoch)+'.pkl')

    #validation using eval.py
    '''
    if epoch % 2 == 0:
        os.makedirs(os.path.dirname(opt.model), exist_ok=True)
        torch.save(feature_extractor.state_dict(), opt.model+str(epoch)+'.pkl')
        if os.path.exists('./tmp.csv'):
            os.remove('./tmp.csv') 
        with open('./tmp.csv', 'w', newline='') as csvfile:
            feature_extractor.eval()
            writer = csv.writer(csvfile)
            writer.writerow(['Id', 'Rank'])
            with torch.no_grad():
                trange = tqdm(enumerate(validDataset), total=len(validDataset), desc='Valid')
                # each training batch only has one video
                for i, (query, cand, qfn, cfn) in trange:
                    query, cand = query.squeeze(0).to(device), cand.squeeze(0).to(device)
                    querys, _ = feature_extractor(query)

                    qnorm = torch.norm(querys, p=2, dim=1, keepdim=True)
                    querys = querys.div(qnorm.expand_as(querys))

                    cands = []
                    num_ = cand.shape[0]//opt.feature_batch 

                    for j in range(num_):
                        tmp = cand[opt.feature_batch*j:opt.feature_batch*(j+1),:]
                        tmp, _ = feature_extractor(tmp)
                        tmpnorm = torch.norm(tmp, p=2, dim=1, keepdim=True)
                        tmp = tmp.div(tmpnorm.expand_as(tmp))                         
                        cands.append(tmp)
                    if cand.shape[0] % opt.feature_batch != 0:
                        tmp = cand[opt.feature_batch*num_:,:]
                        tmp, _ = feature_extractor(tmp)
                        tmpnorm = torch.norm(tmp, p=2, dim=1, keepdim=True)
                        tmp = tmp.div(tmpnorm.expand_as(tmp))
                        cands.append(tmp)
                    cands = torch.cat(cands, 0)
                    output = torch.mm(querys, cands.transpose(0,1))  #(query, cand)
                    rank = torch.argsort(output, dim=1, descending=True)                
                    for i in range(rank.shape[0]):
                        sorted_cands = []
                        for j in range(rank.shape[1]):
                            sorted_cands.append(os.path.basename(cfn[rank[i][j].item()]).split('.')[0])
                        writer.writerow([os.path.basename(qfn[i]).split('.')[0], ' '.join(sorted_cands)])
            trange.set_postfix(num=i)
        os.system('python3 eval.py ./val_GT.json ./tmp.csv')
    '''
