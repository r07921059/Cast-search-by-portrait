![Imgur](https://i.imgur.com/q6DDpTd.png)

## Prerequisites
:ballot_box_with_check: Linux OS

:ballot_box_with_check: Python 3.5+

:ballot_box_with_check: GPU Memory >= 10G

:ballot_box_with_check: CPU Memory >= 64G

:ballot_box_with_check: CPU Shared Memory >= 32G (share memory of dataloader workers)

:ballot_box_with_check: Pytorch 1.0

:ballot_box_with_check: Torchvision 0.2.2

:ballot_box_with_check: Skimage0.15.0

:ballot_box_with_check: Multiprocessing

:ballot_box_with_check: Dlib 19.17.0 (would not used in predict, only for preparing sub-query files :blush: :blush:)
  - If you want to reproduce **all of our steps**, we recommend to installing **GPU version**
  - CPU version install guide
    - (sudo) apt-get install cmake
    - pip3 install (--user) dlib
  - GPU version install guide (build from source)
    - git clone https://github.com/davisking/dlib.git
    - cd dlib
    - mkdir build
    - cd build
    - cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
    - cmake --build .
    - cd ..
    - python3 setup.py install
    
:ballot_box_with_check: Faster R-CNN and Mask R-CNN(would not used in predict, only for preparing sub-query files :blush: :blush:)

   - [**We recommend to following the official installation guide**](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md)
   - pip3 install ninja yacs cython matplotlib tqdm opencv-python
   - cd $INSTALL_DIR
   - git clone https://github.com/cocodataset/cocoapi.git
   - cd cocoapi/PythonAPIe
   - python3 setup.py build_ext install
   - cd $INSTALL_DIR
   - git clone https://github.com/NVIDIA/apex.git
   - cd apex
   - python3 setup.py install --cuda_ext --cpp_ext
   - git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
   - cd maskrcnn-benchmark
   - python3 setup.py build develop

## Files description

:file_folder: model_zoo/ :arrow_right: our model weights

:file_folder: tcnn_mask_subquery/ :arrow_right: our sub-query txt file

:file_folder: dlib_model/ :arrow_right: dlib pre-train model from [**here**](https://github.com/davisking/dlib-models)

:file_folder: maskrcnn/ :arrow_right: using for instance segmentation

:scroll: dataset.py :arrow_right: different ways to get data  

:scroll: eval.py :arrow_right: offer by TAs

:scroll: get_dataset.sh :arrow_right: quick to get datasets

:scroll: model.py :arrow_right: our backbone models  

:scroll: re_ranking.py :arrow_right: re-ranking mechanism from [**here**](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/re_ranking.py)

:scroll: thres_Search_cnn_mask.py :arrow_right: generate files of subquery (like tcnn_mask_subquery directory)

:scroll: triplet_loss.py :arrow_right: hard triplet loss implementation

:scroll: thres_subquery.py :arrow_right: predict test.csv

:scroll: triplet_mixed_task1.py :arrow_right: train our backbones by cross-entropy and triplet loss

## How to reproduce score on kaggle
If you want to reproduce our work as soon as possible, you can just move directly to the **Predicting** part

We **strongly recommend not to runing all steps, it would take a lot of time.** 

### Instance segmentation
(you can skip the step, we have already prepared in the [**google drive**](https://drive.google.com/file/d/1lDjj6N71E3jjhLzwYOjKXrCG71afuN-T/view?fbclid=IwAR16ywkuFKYvKOUAwXmQW8y_nug6AwCK3wxdjjbcBXNs0H1MS4_hpByII4w) :blush: :blush: :smiley: :smiley:)
```bash
cd ./maskrcnn/
python3 mask.py --final_data $1 --mask_output_path $2
```
`$1`: path to serach ‘.jpg’ file that you want to do instance segmenation e.g.  ./final_data  
`$2`: path to output your instance segmentation result e.g.  ./final_data_mask

### Pick sub-query from candidates
(you can skip the step, we have already prepared in the :file_folder: **tcnn_mask_subquery** :blush: :blush: :smiley: :smiley:)
```bash
python3 thres_Search_cnn_mask.py --output_dir $1 --video_dir $2 --mask_dir $3
```
`$1`: destination to output subquery recording file  e.g. ./reproduce_tcnn_mask_subquery  
`$2`: path to where video folders exist e.g. ./final_data/val  
`$3`: path to where masks, which can be produced by instance segmentation, exist e.g. ./final_data_mask/val

### Predicting
(If you **reproduce our subquery by yourself**, you need to specific the subquery directory which you reproduce e.g. ./reproduce_tcnn_mask_subquery
)
```bash
python3 thres_subquery.py --valid_data $1 --output_csv $2 --subquery_log_path $3
```
`$1`: path to which dataset you want to perform predicting e.g. ./final_data/test  
`$2`: path to output csv file **base on exist directorys** e.g. ./reproduce_test.csv  
`$3`: (**optional**) path to subquery directory, we have already set up in default e.g. ./tcnn_mask_subquery 

## How to train by yourself

### Training
```bash
python3 triplet_mixed_task1.py --train_data $1 --valid_data $2 --model $3
```
`$1`: path to training data. e.g. ./final_data/train   
`$2`: path to validation data. e.g. ./final_data/val  
`$3`: model name for saving state dict

## Score on kaggle

| Methods | mAP| epochs | kaggle rank |
| -------- | ---- | ---- | ---- |
| [**densenet121 + mask + subquery + re-ranking**] | **73.06%** | 13 epoch | **Second place**

## Citation
You may cite them in your paper.
```
@article{,
  author    = {Wei-Ting Syu and
               Kuan-Chih Huang and
               Shang-Lun Tsai and
               Wei-Yao Hong}
  title     = {Cast Search by Portrait},
  year      = {Spring, 2019},
}
```

## Related Repos and Papers
1. [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)
2. [WIDER Face and Pedestrian Challenge 2018 Methods and Results](https://arxiv.org/pdf/1902.06854.pdf)
