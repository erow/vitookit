#!/usr/bin/env python
"""
Fine-tuning partial parameters for image classification.
Fine-tune the weights corresponding to the MHSA layer is 10% faster and achieves comparable performance to full fine-tuning. This strategy is particularly useful when adapting large pre-trained models to downstream tasks with limited samples and similar to the pretrainning data.

References:
1. https://arxiv.org/pdf/2203.09795#page=9.30

"""
# from PIL import Image # hack to avoid `CXXABI_1.3.9' not found error

import argparse
import datetime
import re
import time
import timm
import timm.optim
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import json
import os
import math
import sys
from vitookit.datasets.transform import three_augmentation
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
from timm.utils import NativeScaler, accuracy
from timm.data import Mixup
from timm.layers import trunc_normal_

from vitookit.evaluation.eval_cls import get_args_parser, train_one_epoch, evaluate, BatchAugmentation

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
            wandb.init(job_type='finetune',dir=args.output_dir,resume=True if args.resume else False, 
                   config=args.__dict__)
        except:
            pass
    
    if args.ThreeAugment:
        transform = three_augmentation(args.input_size, args.color_jitter, args.src)
    else:
        transform = build_transform(is_train=True, args=args)
    
    dataset_train, args.nb_classes = build_dataset(args=args, is_train=True, trnsfrm=transform)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
        
    print("Load dataset:", dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.ra<2:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_train = BatchAugmentation(sampler_train, repeat=args.ra)
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
    print(f"Built Model ", model)

    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key=args.checkpoint_key, prefix=args.prefix)
    train(args, model,data_loader_train, data_loader_val)

def train(args,model,data_loader_train, data_loader_val):

    model = convert_sync_batchnorm(model)
    model_without_ddp = model
    import torch   
    
    device = torch.device(args.device)
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model.to(device)

    pattern = re.compile(args.param)
    match_params = []
    for name, p in model.named_parameters():
        if pattern.search(name):
            p.requires_grad = True
            match_params.append(name)
        else:
            p.requires_grad = False
    print(f"Matched parameters to train: {match_params}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        linear_scaled_lr = args.blr * eff_batch_size / 256.0
        print("base lr: %.2e" % args.blr )
        print("actual lr: %.2e" % linear_scaled_lr)
        args.lr = linear_scaled_lr
    else:
        print("actual lr: %.2e" % args.lr)
    
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module
        
    if args.opt == 'muon':
        from vitookit.utils.muon import Muon
        # code: https://github.com/KellerJordan/Muon
        # Muon is intended to optimize only the internal ≥2D parameters of a network. Embeddings, classifier heads, and scalar or vector parameters should be optimized using AdamW instead.
        # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
        
        # todo: add layer decay
        muon_params = []
        adamw_params = []
        for name, p in model.named_parameters():
            if 'head' in name or 'fc' in name or 'embed' in name:
                adamw_params.append(p)
            elif p.ndim == 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        print(f"Muon parameters: {len(muon_params)}, AdamW parameters: {len(adamw_params)}")
        if args.opt_betas is None:
            args.opt_betas = (0.9,0.95)
        optimizer = Muon(lr=args.lr, wd=args.weight_decay, momentum=0.95,
                         muon_params=muon_params, adamw_params=adamw_params,
                         adamw_betas=args.opt_betas, adamw_eps=args.opt_eps)
    else:
        optimizer = create_optimizer(args, model_without_ddp, 
                                     filter_bias_and_bn=not args.disable_weight_decay_on_bias_norm)
    
    print('Optimizer: ', optimizer)

    if args.compile:
        model = torch.compile(model)   
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True 
     
    if args.no_amp:
        loss_scaler = None
    else:
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

    print(f"Start training for {args.epochs} epochs from {args.start_epoch}")
    if args.dynamic_resolution:
        from vitookit.datasets.dres import DynamicResolution
        dres = DynamicResolution(args.epochs)
    else:
        dres = None
    start_time = time.time()
    max_accuracy = 0.0
        
    for epoch in range(args.start_epoch, args.epochs):
        if hasattr(data_loader_train,'sampler'):
            # dataloader
            data_loader_train.sampler.set_epoch(epoch)
            if dres: dres(data_loader_train,epoch,False)
        else:
            # ffcv
            if dres: dres(data_loader_train,epoch,True)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,lr_scheduler,
            args.clip_grad,  mixup_fn, accum_iter=args.accum_iter
        )
        
        checkpoint_paths = ['checkpoint.pth']
        
        if epoch%args.ckpt_freq==0 or epoch==args.epochs-1:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
            
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
                        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                        'args': args,
                    }, output_dir / checkpoint_path)
            else:
                misc.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                        'args': args,
                    }, output_dir / "checkpoint.pth")
                
            if wandb.run: wandb.log(log_stats)
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    protocol = os.path.basename(sys.argv[0]).replace('.py', '')
    basename = f"{protocol}-{args.data_set}"
    log_metrics(basename, log_stats, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('param', type=str, help='the training parameters.')
    args = parser.parse_args()
    main(args)
