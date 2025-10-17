#!/usr/bin/env python
"""
Learning rate finder for image classification.


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
from vitookit.models import losses
from vitookit.utils.helper import *
from vitookit.utils import misc
from vitookit.utils.layer_decay import param_groups_lrd
from vitookit.models.build_model import build_model

from vitookit.datasets.build_dataset import build_dataset, build_transform
from timm.layers import (convert_splitbn_model, convert_sync_batchnorm,
                         set_fast_norm)
from torch import nn


from pathlib import Path
from typing import Iterable, Optional

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy
from timm.data import Mixup
from timm.layers import trunc_normal_

from vitookit.evaluation.eval_cls import get_args_parser, train_one_epoch, evaluate, BatchAugmentation

import math
from tqdm.auto import tqdm
import copy
import itertools
from torch.optim import Optimizer
from torch.utils.data import DataLoader
class LRFinder:
    """
    Learning Rate Finder to help find an optimal learning rate for training.

    The LR is increased exponentially from a starting value to an ending value
    for a certain number of iterations. The loss is recorded at each step,
    and a plot of loss vs. LR can be used to determine the optimal LR.
    """
    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion, mixup_fn,device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.mixup_fn = mixup_fn
        self.device = device
        self.history = {"lr": [], "loss": []}
        
        # Save initial state
        self._initial_state = copy.deepcopy(model.state_dict())
        self._initial_optim_state = copy.deepcopy(optimizer.state_dict())

    @gin.configurable
    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
    ):
        """
        Performs the LR range test.

        Args:
            train_loader (DataLoader): The training data loader.
            start_lr (float): The starting learning rate.
            end_lr (float): The ending learning rate.
            num_iter (int): The number of iterations to perform the test for.
            smooth_f (float): Smoothing factor for the loss.
        """
        self.history = {"lr": [], "loss": []}
        self.model.train()
        
        # Reset model and optimizer to initial state
        self.model.load_state_dict(self._initial_state)
        self.optimizer.load_state_dict(self._initial_optim_state)

        # Calculate the multiplication factor for LR update
        gamma = (end_lr / start_lr) ** (1 / (num_iter - 1))
        lr = start_lr
        
        # Set the initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        avg_loss = 0.0
        best_loss = float('inf')
        
        # Use tqdm for a progress bar
        pbar = tqdm(total=num_iter, desc="LR Finder")
        
        # Use itertools.cycle to handle cases where loader has fewer batches than num_iter
        data_iterator = itertools.cycle(train_loader)

        for i in range(num_iter):
            try:
                inputs, labels = next(data_iterator)
            except StopIteration:
                # This should not happen with itertools.cycle
                break

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if self.mixup_fn is not None:
                inputs, labels = self.mixup_fn(inputs, labels)
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update learning rate
            lr *= gamma
            for param_group in self.optimizer.param_groups:
                if 'lr_scale' in param_group:
                    param_group['lr'] = lr * param_group['lr_scale']
                else:
                    param_group['lr'] = lr

            # Record and smooth the loss
            current_loss = loss.item()
            if i == 0:
                avg_loss = current_loss
            else:
                avg_loss = (1 - smooth_f) * avg_loss + smooth_f * current_loss
            
            # Use a smoothed loss for the plot
            smoothed_loss = avg_loss / (1 - (1 - smooth_f) ** (i + 1))
            
            # Stop if the loss explodes
            if smoothed_loss > 4 * best_loss and i > 10:
                print("\nLoss exploded. Stopping the test.")
                break
                
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            self.history["lr"].append(lr)
            self.history["loss"].append(smoothed_loss)
            pbar.update(1)
            pbar.set_postfix({"loss": f"{smoothed_loss:.4f}", "lr": f"{lr:.2e}"})

        pbar.close()
        print("LR Finder test complete.")
        
        # Restore model to its initial state
        self.model.load_state_dict(self._initial_state)
        self.optimizer.load_state_dict(self._initial_optim_state)

    def plot(self, output_dir=None, skip_start: int = 0, skip_end: int = 5):
        """
        Plots the loss vs. learning rate.

        Args:
            skip_start (int): Number of initial iterations to skip in the plot.
            skip_end (int): Number of final iterations to skip in the plot.
        """
        if not self.history["lr"]:
            print("Please run range_test() first.")
            return

        lrs = self.history["lr"][skip_start:-skip_end]
        losses = self.history["loss"][skip_start:-skip_end]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.yticks(rotation=45)
        plt.title("Learning Rate Finder")
        plt.grid(True, which="both", ls="--")
        # add a twin axis to show the delta loss
        ax2 = plt.gca().twinx()
        delta_losses = [j - i for i, j in zip(losses[:-1], losses[1:])]
        ax2.plot(lrs[1:], delta_losses, color='orange', alpha=0.5)
        ax2.set_ylabel("Delta Loss", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        plt.show()
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, "lr_finder_plot.png"))
            print(f"LR finder plot saved to {os.path.join(output_dir, 'lr_finder_plot.png')}")
        
def main(args):
    misc.init_distributed_mode(args)
    # fix the seed for reproducibility
    misc.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print(args)
    post_args(args)

    
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

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))
    

    output_dir = Path(args.output_dir) if args.output_dir else Path('.')

    lr_finder = LRFinder(model, optimizer, criterion, mixup_fn, device)
    lr_finder.range_test(
        train_loader=data_loader_train,
    )
    lr_finder.plot(output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
