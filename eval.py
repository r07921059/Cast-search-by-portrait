import numpy
import os
import os.path as osp
import json
from random import shuffle
import argparse


def parse_submission(submission_file):
    with open(submission_file) as f:
        lines = f.readlines()
    submission = {}
    for line in lines[1:]:
        words = line.strip().split(',')
        if len(words) != 2:
            print('Format Error!')
            return None
        key = words[0].strip()
        ret = words[1].strip().split()
        unique_ret = []
        appeared_set = set()
        for x in ret:
            if x not in appeared_set:
                unique_ret.append(x)
                appeared_set.add(x)
        submission[key] = unique_ret
    return submission


def read_gt(gt_file):
    with open(gt_file) as f:
        data = json.load(f)
    gt_dict = {}
    for key, value in data.items():
        gt_dict[key] = set(value)
    return gt_dict


def get_AP(gt_set, ret_list):
    hit = 0
    AP = 0.0
    for k, x in enumerate(ret_list):
        if x in gt_set:
            hit += 1
            prec = hit / (k+1)
            AP += prec
    AP /= len(gt_set)
    return AP


def get_mAP(gt_dict, ret_dict):
    mAP = 0.0
    query_num = len(gt_dict.keys())
    for key, gt_set in gt_dict.items():
        if ret_dict.get(key) is None:
            AP = 0
        else:
            AP = get_AP(gt_set, ret_dict[key])
        mAP += AP
    mAP /= query_num
    return mAP


def eval(submission_file, gt_file):
    gt_dict = read_gt(gt_file)
    submission = parse_submission(submission_file)
    print(len(gt_dict))
    print(len(submission))
    mAP = get_mAP(gt_dict, submission)
    print('mAP: {:.4f}'.format(mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', type=str)
    parser.add_argument('submission', type=str)
    args = parser.parse_args()

    eval(args.submission, args.gt)

