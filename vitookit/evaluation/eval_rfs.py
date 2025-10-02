#!/usr/bin/env python
"""
Example:
vitrun  --nproc_per_node=8 eval_attr.py --data_location ~/data/RIVAL10/ --data_set rival10   --output_dir outputs/semi

"""
from PIL import Image # hack to avoid `CXXABI_1.3.9' not found error

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from vitookit.datasets.ffcv_transform import *

from vitookit.utils.helper import post_args, load_pretrained_weights
from vitookit.utils import misc
from vitookit.models.build_model import build_model
from vitookit.datasets.build_dataset import build_transform
from vitookit.evaluation.eval_cls import get_args_parser, NativeScaler, convert_sync_batchnorm, create_optimizer, create_scheduler, train_one_epoch,  restart_from_checkpoint, log_metrics, os, sys, json, Path
import wandb

def add_noise(x, mask, noise_level=0.1,):
    return (x + noise_level * torch.randn_like(x) * mask).clamp(0,1)

def tfms(x):
    device = x.device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
    return (x - mean) / std

@torch.no_grad()
def rfs_sensitivity(model, img, mask, sigma,return_all=False):
    x_fg = add_noise(img, mask, sigma)
    x_bg = add_noise(img, 1-mask, sigma)
    anchor = model(tfms(img)).data
    fg_z = model(tfms(x_fg)).data
    bg_z = model(tfms(x_bg)).data
    fg_sim = torch.nn.functional.cosine_similarity(anchor, fg_z)
    bg_sim = torch.nn.functional.cosine_similarity(anchor, bg_z)
    a_sim = (fg_sim + bg_sim)/2
    rfs = (bg_sim - fg_sim) / (torch.min(a_sim, 1-a_sim) + 1e-6) / 2
    if return_all:
        return rfs, fg_sim, bg_sim
    return rfs

from vitookit.datasets import rival10


class RIVAL10Mask(rival10.LocalRIVAL10):
    def __getitem__(self, i):
        out = super().__getitem__(i)
        return out['img'], out['merged_mask']

def main(args):
    misc.init_distributed_mode(args)
    # fix the seed for reproducibility
    misc.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print(args)
    post_args(args)

    # logging system
    if args.output_dir and misc.is_main_process():
        try:
            wandb.init(job_type='attr',dir=args.output_dir,resume=True, 
                   config=args.__dict__)
        except:
            pass
    
    
    transform_train = build_transform(True,args)
    transform_val = build_transform(False,args)
    
    print("Transforms: ", transform_train, transform_val)

    if args.data_set == 'rival10':
        dataset_train = RIVAL10Mask(args.data_location, train=True)
        dataset_val = RIVAL10Mask(args.data_location, train=False)
        args.nb_classes = 18 # number of attributes    
    else:
        raise NotImplementedError
        
    print("Load dataset:", dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
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

    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key=args.checkpoint_key, prefix=args.prefix)
    model.requires_grad_(False)
    if hasattr(model,'fc'):
        model.fc.requires_grad_(True)
    else:
        model.head.requires_grad_(True)
    # print(f"Built Model ", model)
    log = evaluate(args, data_loader_train, model,)
    print(log)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'log.txt', 'a') as f:
        json.dump(log, f)
        f.write('\n')


@torch.no_grad()
def evaluate(args, data_loader, model):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    for images, masks in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True, dtype=torch.long)

        # compute output
        with torch.cuda.amp.autocast():
            rfs = rfs_sensitivity(model, images, masks, args.sigma).mean()
        batch_size = images.shape[0]
        metric_logger.meters['rfs'].update(rfs.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes() 
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Relative Foreground Sensitivity', parents=[get_args_parser()])
    parser.add_argument("--sigma", default=0.6, type=float, help="Noise level")
    args = parser.parse_args()
    main(args)
