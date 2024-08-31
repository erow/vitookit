#!/usr/bin/env python


"""
Example:
vitrun  --nproc_per_node=3 eval_cls_ffcv.py --train_path $train_path --val_path $val_path  --gin VisionTransformer.global_pool='\"avg\"'  -w wandb:dlib/EfficientSSL/xsa4wubh  --batch_size 360 --output_dir outputs/cls

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
import wandb

from ffcv import Loader
from ffcv.loader import OrderOption
from vitookit.evaluation.eval_cls import get_args_parser,train
from timm.layers import (convert_splitbn_model, convert_sync_batchnorm,
                         set_fast_norm)

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
            wandb.init(job_type='finetune',dir=args.output_dir,resume=True, 
                   config=args.__dict__)
        except:
            pass
    
    order = OrderOption.RANDOM if args.distributed else OrderOption.QUASI_RANDOM
    data_loader_train =  Loader(args.train_path, pipelines=ThreeAugmentPipeline(),batches_ahead=1,
                        batch_size=args.batch_size, num_workers=args.num_workers, 
                        order=order, distributed=args.distributed,seed=args.seed)
    

    data_loader_val =  Loader(args.val_path, pipelines=ValPipeline(),
                        batch_size=args.batch_size, num_workers=args.num_workers, batches_ahead=1,
                        distributed=args.distributed,seed=args.seed)
    print("Load dataset:", data_loader_train)

    
    model = build_model(num_classes=args.nb_classes)
    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key=args.checkpoint_key, prefix=args.prefix)
    print(f"Built Model ", model)
    train(args, model,data_loader_train, data_loader_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    parser.add_argument("--train_path", type=str, required=True, help="The path of ffcv file.")
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--nb_classes", default=1000, type=int, help="The number of classes in the dataset.")
    args = parser.parse_args()
    main(args)
