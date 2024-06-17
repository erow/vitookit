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
import json
import os
import math
import sys
import copy
import scipy.io as scio
from vitookit.datasets.ffcv_transform import *
from vitookit.evaluation.eval_cls import evaluate
from vitookit.utils.helper import *
from vitookit.utils import misc
from vitookit.models.build_model import build_model
from vitookit.datasets.dres import DynamicResolution
import wandb

from pathlib import Path
from typing import Iterable, Optional
from torch.nn import functional as F


from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from timm.data import Mixup
from timm.layers import trunc_normal_

from ffcv import Loader
from ffcv.loader import OrderOption

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--ckpt_freq', default=5, type=int)
    parser.add_argument("--dynamic_resolution", default=False, action="store_true", help="Use dynamic resolution.")

    # Model parameters
    parser.add_argument("--compile", action='store_true', default=False, help="compile model with PyTorch 2.0")
    parser.add_argument("--prefix", default=None, type=str, help="prefix of the model name")
    
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('-w', '--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default=None, type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')


    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--layer_decay', type=float, default=None)
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--ThreeAugment', action='store_true', default=True) #3augment
    parser.add_argument('--src',action='store_true', default=False, 
                        help="Use Simple Random Crop (SRC) or Random Resized Crop (RRC). Use SRC when there is less risk of overfitting, such as on ImageNet-21k.")
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=None, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--disable_weight_decay_on_bias_norm', action='store_true', default=False)
    parser.add_argument('--init_scale', default=1.0, type=float)

    # Dataset parameters
    parser.add_argument('--train_path', type=str, required=True, help='path to train dataset')
    parser.add_argument('--val_path', type=str, required=True, help='path to test dataset')
    parser.add_argument('--nb_classes', type=int, default=1000, help='number of classes')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)

    # configure
    parser.add_argument('--cfgs', nargs='+', default=[],
                        help='<Required> Config files *.gin.', required=False)
    parser.add_argument('--gin', nargs='+', 
                        help='Overrides config values. e.g. --gin "section.option=value"')

    # distributed training parameters
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,lr_scheduler, max_norm: float = 0,
                     mixup_fn: Optional[Mixup] = None,
                     accum_iter = 1
                    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = max(len(data_loader)//20,20)
    
    for itr,(samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        lr_scheduler.step(epoch+itr/len(data_loader))
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    need_update=(itr + 1) % accum_iter == 0)
        if (itr + 1) % accum_iter == 0:
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        # log metrics
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        if wandb.run: 
            wandb.log({'train/loss':loss})
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    misc.init_distributed_mode(args)
    post_args(args)
    print("args: ", args)
    print("configure: ", gin.config_str())
    import torch
    device = torch.device(args.device)

    # fix the seed for reproducibility
    misc.fix_random_seeds(args.seed)

    cudnn.benchmark = True
    
    
    order = OrderOption.RANDOM if args.distributed else OrderOption.QUASI_RANDOM
    data_loader_train =  Loader(args.train_path, pipelines=ThreeAugmentPipeline(),batches_ahead=1,
                        batch_size=args.batch_size, num_workers=args.num_workers, 
                        order=order, distributed=args.distributed,seed=args.seed)
    

    data_loader_val =  Loader(args.val_path, pipelines=ValPipeline(),
                        batch_size=args.batch_size, num_workers=args.num_workers, batches_ahead=1,
                        distributed=args.distributed,seed=args.seed)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    
    print(f"Model  built.")
    # load weights to evaluate
    
    model = build_model(num_classes=args.nb_classes,drop_path_rate=args.drop_path,)
    if args.output_dir and misc.is_main_process():
        try:
            wandb.init(job_type='finetune',dir=args.output_dir,resume=True, 
                   config=args.__dict__)
        except:
            pass
    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key=args.checkpoint_key, prefix=args.prefix)
    if args.compile:
        model = torch.compile(model)    
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True 
    # trunc_normal_(model.head.weight, std=2e-5)
    
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    linear_scaled_lr = args.lr * eff_batch_size / 256.0
    
    print("base lr: %.2e" % args.lr )
    print("actual lr: %.2e" % linear_scaled_lr)
    args.lr = linear_scaled_lr
    
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))
    

    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.resume:        
        run_variables={"args":dict(),"epoch":0}
        restart_from_checkpoint(args.resume,
                                optimizer=optimizer,
                                model=model_without_ddp,
                                scaler=loss_scaler,
                                run_variables=run_variables)
        # args = run_variables['args']
        args.start_epoch = run_variables["epoch"] + 1

    if args.eval:
        assert args.world_size == 1
        test_stats, preds = evaluate(data_loader_val, model_without_ddp, device,True)
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        if args.output_dir and misc.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(test_stats) + "\n")
            with (output_dir / "preds.txt").open("w") as f:
                f.write(json.dumps(preds.tolist()) + "\n")
        exit(0)

    print(f"Start training for {args.epochs} epochs from {args.start_epoch}")
    
    if args.dynamic_resolution:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        dres = DynamicResolution()    
    else:
        dres = None
        
    start_time = time.time()
    max_accuracy = 0.0
        
    for epoch in range(args.start_epoch, args.epochs):
        if dres:
            dres(data_loader_train,epoch,True)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,lr_scheduler,
            args.clip_grad,  mixup_fn, accum_iter=args.accum_iter
        )
        
        checkpoint_paths = ['checkpoint.pth']
        
        if epoch%args.ckpt_freq==0 or epoch==args.epochs-1:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on test images: {test_stats['acc1']:.1f}%")
            
            if (test_stats["acc1"] >= max_accuracy):
                # always only save best checkpoint till now
                checkpoint_paths += [ 'checkpoint_best.pth']
                
        
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                        **{f'test/{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            
        # only save checkpoint on rank 0
        if output_dir and misc.is_main_process():
            if epoch%args.ckpt_freq==0 or epoch==args.epochs-1:
                for checkpoint_path in checkpoint_paths:
                    misc.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, output_dir / checkpoint_path)
                
            if wandb.run: wandb.log(log_stats)
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    basename = os.path.basename(__file__)
    log_metrics(basename, log_stats, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
