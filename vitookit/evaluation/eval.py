#!/usr/bin/env python
# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from DEiT library:
https://github.com/facebookresearch/deit/blob/main/main.py
"""
# from PIL import Image # hack to avoid `CXXABI_1.3.9' not found error

import argparse
import datetime
import time
from sklearn.metrics import f1_score
import timm
import timm.optim
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import json
import os
import math
import sys
from vitookit.utils.helper import *
from vitookit.utils import misc
from vitookit.utils.layer_decay import param_groups_lrd
from vitookit.models.build_model import build_model

from vitookit.datasets.build_dataset import build_dataset, build_transform
from timm.layers import (convert_splitbn_model, convert_sync_batchnorm,
                         set_fast_norm)

import wandb


from pathlib import Path
from typing import Iterable, Optional

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy, metrics
from timm.data import Mixup
from timm.layers import trunc_normal_


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--ckpt_freq', default=5, type=int)

    # Model parameters
    parser.add_argument("--model", default='vit_base_patch16_224', type=str, help="model name")
    parser.add_argument("--compile", action='store_true', default=False, help="compile model with PyTorch 2.0")
    parser.add_argument("--prefix", default=None, type=str, help="prefix of the model name")
    
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('-w', '--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default=None, type=str, help='Key to use in the checkpoint (example: "teacher")')


    parser.add_argument('--dump_features', action='store_true', default=False)
    # Augmentation parameters
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--ThreeAugment', action='store_true', default=False) #3augment
    parser.add_argument('--src',action='store_true', default=False, 
                        help="Use Simple Random Crop (SRC) or Random Resized Crop (RRC). Use SRC when there is less risk of overfitting, such as on ImageNet-21k.")


    # Dataset parameters
    parser.add_argument('--data_location', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IN1K', 
                            type=str, help='ImageNet dataset path')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # configure
    parser.add_argument('--cfgs', nargs='+', default=[],
                        help='<Required> Config files *.gin.', required=False)
    parser.add_argument('--gin', nargs='+', 
                        help='Overrides config values. e.g. --gin "section.option=value"')

    # distributed training parameters
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser

def extract_features(model, data_loader):
    metric_logger = misc.MetricLogger(delimiter="  ")
    features = []
    labels = []
    for samples, labels_c in metric_logger.log_every(data_loader, 50):
        samples = samples.cuda(non_blocking=True)
        labels_c = labels_c.cuda(non_blocking=True)
        feats = model(samples).clone()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        feats_list = [torch.zeros_like(feats) for _ in range(dist.get_world_size())]
        labels_list = [torch.zeros_like(labels_c) for _ in range(dist.get_world_size())]
        dist.all_gather(feats_list, feats)
        dist.all_gather(labels_list, labels_c)
        feats = torch.cat(feats_list)
        labels_c = torch.cat(labels_list)
        
        if misc.is_main_process():
            features.append(feats.cpu())
            labels.append(labels_c.cpu())
    dist.barrier()
    if misc.is_main_process():
        features = torch.cat(features)
        labels = torch.cat(labels)
        print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        return features, labels


def main(args):
    misc.init_distributed_mode(args)
    # fix the seed for reproducibility
    misc.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print(args)
    post_args(args)
            
    transform = build_transform(is_train=False, args=args)
    
    dataset_val, args.nb_classes = build_dataset(args=args, is_train=False, trnsfrm=transform)
    
        
    print("Load dataset:", dataset_val)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )    

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    # load weights to evaluate
    
    model = build_model(args.model, num_classes=args.nb_classes)
    print(f"Built Model ", model)

    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key=args.checkpoint_key, prefix=args.prefix)
    model = convert_sync_batchnorm(model)
    model.eval()
    model.requires_grad_(False)
    model.cuda()
        
    features, labels = extract_features(model, data_loader_val)
    if misc.is_main_process():
        if args.dump_features and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(features, os.path.join(args.output_dir, "features.pth"))
            torch.save(labels, os.path.join(args.output_dir, "labels.pth"))
        
        # calculate top-1 accuracy
        top1 = accuracy(features, labels, topk=(1,))
        print(f"Top-1 accuracy: {top1[0].item()}")
        
        # calculate top-5 accuracy
        top5 = accuracy(features, labels, topk=(5,))
        print(f"Top-5 accuracy: {top5[0].item()}")
        
        # calculate f1 score
        f1 = f1_score(labels, features.argmax(1), average='macro')
        print(f"F1 score: {f1}")
        
        # confusion matrix
        res_confusion_matrix = metrics.confusion_matrix(features, labels)
        if len(res_confusion_matrix) < 20:
            print(f"Confusion matrix: \n{res_confusion_matrix}")
        if args.output_dir:
            np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), res_confusion_matrix)
    
    dist.barrier()
    dist.destroy_process_group()
        
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
