import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from io import BytesIO
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from dlcv_predictor_new import COCODemo
import cv2
import sys
import os
import glob
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default='./dlcv_e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml', help="config_file")
parser.add_argument("--final_data", type=str, default='./final_data/', help="e.g. ./final_data")
parser.add_argument("--mask_output_path", type=str, default='./final_data_mask/', help="e.g. ./final_data_mask")
opt = parser.parse_args()



final_dir=opt.final_data.strip('/')
final_dir_len=len(final_dir)
if os.path.isabs(opt.final_data):
    final_dir_len +=1  
config_file=opt.config_file
# update the config options with the config file
cfg.merge_from_file(config_file)

print('gathering data')
img_path_list=sorted(glob.glob(os.path.join(opt.final_data,'**','*.jpg'),recursive=True))


print('init maskrcnn')
coco_demo = COCODemo(
    cfg,
    min_image_size=600,
    confidence_threshold=0.7,
)

for i,img_path in enumerate(img_path_list):
    s_time=time.time()
    print(img_path)
    mask_path=os.path.join(opt.mask_output_path,img_path[final_dir_len+1:])
    mask_path=mask_path.replace('.jpg','_mask.png') 
    print(mask_path)
    #pred_path=img_path.replace('final_data','final_data_pred_jpg')
    #pred_path=pred_path.replace('.jpg','_pred.jpg') 
    

    if not os.path.exists(os.path.dirname(mask_path)):
        os.makedirs(os.path.dirname(mask_path))

    #if not os.path.exists(os.path.dirname(pred_path)):
    #    os.makedirs(os.path.dirname(pred_path))

    image=cv2.imread(img_path,1)

    predictions,mask = coco_demo.run_on_opencv_image(image)
    
    #print(mask_path)
    #print(pred_path)
    cv2.imwrite(mask_path,mask)
    #cv2.imwrite(pred_path,predictions)
    e_time=time.time()
    print('{}/{} time: {} sec'.format(i,len(img_path_list),e_time-s_time))

