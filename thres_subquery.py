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
from re_ranking import *
# sys.path.append("./Human-Segmentation-PyTorch")
# from models import UNet
# from torch.nn import functional as F


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().to(device)  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=3, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=tuple, default=(336,192), help="size of each image dimension") #(384,192) #(368,192)
parser.add_argument("--valid_data", type=str, default='final_data/val', help="Path to video folder, e.g. ./final_data/val")
parser.add_argument("--feature_batch", type=int, default=128, help="size of the batches")
parser.add_argument("--model", type=str, default='./model_zoo/REID_trip_336_192_tmixv.pkl13.pkl', help="model path")
parser.add_argument("--output_csv", type=str, default='./reproduce_test.csv', help="output csv path")
parser.add_argument("--subquery_log_path", type=str, default='./tcnn_mask_subquery', help="subquery_log_path")
# parser.add_argument("--model", type=str, default='./final_model/REID5.pkl', help="model path")
opt = parser.parse_args()
print(opt)

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

transform_val_list = [
        transforms.Resize(size=opt.img_size,interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

validDataset = SubqueryDataset(
                subquery= opt.subquery_log_path,
                queryNum = opt.subquery_log_path,
                dataPath=opt.valid_data,
                transformImage=transforms.Compose(transform_val_list))

validDataloader = torch.utils.data.DataLoader(
    validDataset,
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# Initialize model && to gpu
feature_extractor = ft_net_test().to(device)

loaded_dict=torch.load(opt.model)
state_dict = feature_extractor.state_dict()
for key in loaded_dict.keys():
    if key in state_dict.keys():
        state_dict[key]=loaded_dict[key]

feature_extractor.load_state_dict(state_dict)
# seg = UNet(backbone="resnet18", num_classes=2).to(device)
# trained_dict = torch.load('./segment_model/UNet.pkl', map_location="cpu")['state_dict']
# seg.load_state_dict(trained_dict, strict=False)


if os.path.exists(opt.output_csv):
    os.remove(opt.output_csv) 
with open(opt.output_csv, 'w', newline='') as csvfile:
    feature_extractor.eval()
    # seg.eval()
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Rank'])
    with torch.no_grad():
        trange = tqdm(enumerate(validDataloader), total=len(validDataloader), desc='Valid')
        # each training batch only has one video
        for i, (query, cand, qfn, cfn, v_num) in trange:
            query, cand = query.squeeze(0), cand.squeeze(0)
            # mask = seg(query)
            # mask = F.interpolate(mask, size=opt.img_size, mode='bilinear', align_corners=True)
            # mask = F.softmax(mask, dim=1)
            # mask = (mask > 0.5).type(torch.cuda.FloatTensor)
            # query = mask[:,1,:]*query.transpose(0,1)
            # query = query.transpose(0,1)

            # querys = []
            # num_ = query.shape[0]//opt.feature_batch
            # for j in range(num_):
            #     tmp = query[opt.feature_batch*j:opt.feature_batch*(j+1),:]
            #     tmp2 = fliplr(tmp)
            #     tmp, _ = feature_extractor(tmp)
            #     tmp2,_ = feature_extractor(tmp2)
            #     tmp = tmp + tmp2
            #     tmpnorm = torch.norm(tmp, p=2, dim=1, keepdim=True)
            #     tmp = tmp.div(tmpnorm.expand_as(tmp))                         
            #     querys.append(tmp)
            # if query.shape[0] % opt.feature_batch != 0:
            #     tmp = query[opt.feature_batch*num_:,:]
            #     tmp2 = fliplr(tmp)
            #     tmp, _ = feature_extractor(tmp)
            #     tmp2,_ = feature_extractor(tmp2)
            #     tmp = tmp + tmp2
            #     tmpnorm = torch.norm(tmp, p=2, dim=1, keepdim=True)
            #     tmp = tmp.div(tmpnorm.expand_as(tmp))
            #     querys.append(tmp)
            # querys = torch.cat(querys, 0)                            
            #query2 = fliplr(query)                
            #querys, _ = feature_extractor(query)
            #querys2, _ = feature_extractor(query2)
            #querys = querys + querys2
            #qnorm = torch.norm(querys, p=2, dim=1, keepdim=True)
            #querys = querys.div(qnorm.expand_as(querys))
            q_batch=32
            querys=[]
            num_ = query.shape[0]//q_batch 
            for j in range(num_):
                tmp = query[q_batch*j:q_batch*(j+1),:].to(device)
                tmp2 = fliplr(tmp)
                tmp = feature_extractor(tmp)
                tmp2 = feature_extractor(tmp2)
                tmp = tmp.cpu().detach()
                tmp2 = tmp2.cpu().detach()
                
                tmp = tmp + tmp2
                tmpnorm = torch.norm(tmp, p=2, dim=1, keepdim=True)
                tmp = tmp.div(tmpnorm.expand_as(tmp))
                querys.append(tmp)
            if  query.shape[0] % q_batch != 0:
                tmp = query[q_batch*num_:,:].to(device)
                #print(tmp)
                tmp2 = fliplr(tmp)
                tmp = feature_extractor(tmp)
                tmp2 = feature_extractor(tmp2)
                tmp = tmp.cpu().detach()
                tmp2 = tmp2.cpu().detach()
                tmp = tmp + tmp2
                if tmp.dim()==1:
                    tmp=tmp.unsqueeze(0)
                tmpnorm = torch.norm(tmp, p=2, dim=1, keepdim=True)
                tmp = tmp.div(tmpnorm.expand_as(tmp))
                querys.append(tmp)
            querys = torch.cat(querys, 0)

            cands = []
            num_ = cand.shape[0]//opt.feature_batch 

            for j in range(num_):
                tmp = cand[opt.feature_batch*j:opt.feature_batch*(j+1),:].to(device)
                tmp2 = fliplr(tmp)
                tmp = feature_extractor(tmp)
                tmp2 = feature_extractor(tmp2)
                tmp = tmp.cpu().detach()
                tmp2 = tmp2.cpu().detach()
                tmp = tmp + tmp2
                tmpnorm = torch.norm(tmp, p=2, dim=1, keepdim=True)
                tmp = tmp.div(tmpnorm.expand_as(tmp))                         
                cands.append(tmp)
            if cand.shape[0] % opt.feature_batch != 0:
                tmp = cand[opt.feature_batch*num_:,:].to(device)
                tmp2 = fliplr(tmp)
                tmp = feature_extractor(tmp)
                tmp2 = feature_extractor(tmp2)
                tmp = tmp.cpu().detach()
                tmp2 = tmp2.cpu().detach()
                tmp = tmp + tmp2
                tmpnorm = torch.norm(tmp, p=2, dim=1, keepdim=True)
                tmp = tmp.div(tmpnorm.expand_as(tmp))
                cands.append(tmp)
            cands = torch.cat(cands, 0)
            output = torch.mm(querys, cands.transpose(0,1))  #(query, cand)
            final_query = []
            # q_num = 8
            # print(v_num)
            for k in range(0, output.shape[0], 40):
                #tmp = output[k:k+8,:]
                q_g = output[k:k+int(v_num[k][0]),:].cpu().numpy()
                q_q = torch.mm(querys[k:k+int(v_num[k][0]),:], querys[k:k+int(v_num[k][0]),:].transpose(0,1)).cpu().numpy()
                g_g = torch.mm(cands, cands.transpose(0,1)).cpu().numpy()
                tmp = re_ranking(q_g, q_q, g_g,25,6,0.05)
                tmp = torch.from_numpy(tmp)
                # tmp = torch.nn.functional.softmax(tmp, dim=1)
                final_query.append(torch.mean(tmp, dim=0))
            output = torch.stack(final_query,0)

            # sf_output = torch.nn.functional.softmax(output, dim=1)
            # sf_output, _ = torch.sort(sf_output, dim=1)
            # sf_output = sf_output[:,:5]
            # rank = torch.argsort(output, dim=1, descending=True)
            # re_rank_output =  torch.zeros((output.shape[0],output.shape[1])).to(device)
            # for t in range(5): 
            #     first_query = []
            #     for k in range(rank.shape[0]):
            #         first_query.append(sf_output[k][t]*cands[rank[k][t]])
            #     first_query = torch.stack(first_query, 0)
            #     output_first = torch.mm(first_query, cands.transpose(0,1))  #(query, cand)
            #     re_rank_output += output_first

            # rank = torch.argsort(re_rank_output, dim=1, descending=True)
            # rank = torch.argsort(output, dim=1, descending=True)
            rank = torch.argsort(output, dim=1)

            for k in range(rank.shape[0]):
                sorted_cands = []
                for j in range(rank.shape[1]):
                    sorted_cands.append(os.path.basename(cfn[rank[k][j].item()][0]).split('.')[0])
                writer.writerow([os.path.basename(qfn[k][0]).split('.')[0], ' '.join(sorted_cands)])
            trange.set_postfix(num=i)
#os.system('python3 eval.py ./val_GT.json {}'.format(opt.output_csv))


