#!/usr/bin/env python


"""
Example:
vitrun  --nproc_per_node=3 eval_cls_ffcv.py --train_path $train_path --val_path $val_path  --gin VisionTransformer.global_pool='\"avg\"'  -w wandb:dlib/EfficientSSL/xsa4wubh  --batch_size 360 --output_dir outputs/cls

"""
import math
import sys
from typing import Iterable, Optional
from PIL import Image # hack to avoid `CXXABI_1.3.9' not found error

import argparse
import datetime
import numpy as np
import time
import timm
from timm.data.mixup import Mixup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from vitookit.models.losses import MarginHead
from vitookit.utils.helper import post_args, load_pretrained_weights
from vitookit.utils import misc
from vitookit.models.build_model import build_model
import wandb

from vitookit.evaluation.eval_cls import get_args_parser,train
from vitookit.evaluation import eval_cls
from timm.layers import (convert_splitbn_model, convert_sync_batchnorm,
                         set_fast_norm)

class MarginModel(nn.Module):
    def __init__(self, backbone, head):
        super(MarginModel, self).__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x,*args,**kwargs):
        x = self.backbone(x)
        x = self.head(x, *args, **kwargs)
        return x


def calculate_angle(vec1, vec2):
    """
    Calculates the angle in radians between two vectors.
    Args:
        vec1 (torch.Tensor): The first vector.
        vec2 (torch.Tensor): The second vector.
    Returns:
        torch.Tensor: The angle in radians.
    """
    # Normalize the vectors to unit length
    vec1_norm = F.normalize(vec1, p=2, dim=0)
    vec2_norm = F.normalize(vec2, p=2, dim=0)

    # Calculate the cosine similarity (dot product of normalized vectors)
    # Clamp the value to [-1.0, 1.0] to avoid numerical errors with acos
    cosine_similarity = torch.dot(vec1_norm, vec2_norm).clamp(-1.0, 1.0)

    # Calculate the angle in radians
    angle = torch.acos(cosine_similarity)
    return angle

def calculate_metrics(weights, features, labels):
    """
    Calculates WC-Intra, W-Inter, and C-Inter metrics.
    Args:
        weights (torch.Tensor): The weight matrix of the last layer.
                                Shape: (embedding_dim, num_classes)
        features (torch.Tensor): The embedded features for a set of samples.
                                 Shape: (num_samples, embedding_dim)
        labels (torch.Tensor): The ground truth labels for the features.
                               Shape: (num_samples,)
    Returns:
        tuple: A tuple containing the values for (wc_intra, w_inter, c_inter).
    """
    embedding_dim, num_classes = weights.shape
    assert num_classes == labels.max() + 1

    # --- 1. Calculate Class Centers ---
    class_centers = torch.zeros(num_classes, embedding_dim, device=features.device)
    for i in range(num_classes):
        # Find features belonging to the current class
        class_features = features[labels == i]
        if len(class_features) > 0:
            class_centers[i] = class_features.mean(dim=0)

    # --- 2. Calculate WC-Intra (Intra-class Compactness) ---
    # Measures the average angle between each class's weight vector and its feature center.
    # A smaller value indicates better intra-class compactness.
    total_intra_angle = 0.0
    for i in range(num_classes):
        w_i = weights[:, i]
        center_i = class_centers[i]
        total_intra_angle += calculate_angle(w_i, center_i)
    wc_intra = total_intra_angle / num_classes

    # --- 3. Calculate W-Inter (Inter-class Weight Separability) ---
    # Measures the average angle between weight vectors of different classes.
    # A larger value indicates better separability of class boundaries.
    total_w_inter_angle = 0.0
    count = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            w_i = weights[:, i]
            w_j = weights[:, j]
            total_w_inter_angle += calculate_angle(w_i, w_j)
            count += 1
    w_inter = total_w_inter_angle / count if count > 0 else 0.0


    # --- 4. Calculate C-Inter (Inter-class Center Separability) ---
    # A larger value indicates better separability.
    total_c_inter_angle = 0.0
    count = 0
    for j in range(num_classes):
        for i in range(num_classes):
            if i == j:
                continue
            center_j = class_centers[j]
            w_i = weights[:, i]
            total_c_inter_angle += calculate_angle(center_j, w_i)
            count += 1
    c_inter = total_c_inter_angle / count if count > 0 else 0.0

    return wc_intra.item(), w_inter.item(), c_inter.item()


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    features = []
    targets = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        target = target.to(device, non_blocking=True, dtype=torch.long)

        # compute output
        # with torch.cuda.amp.autocast():
        output, z = model(images, labels=target, return_z=True)
        loss = criterion(output, target)
        features.append(z.cpu())
        targets.append(target.cpu())

        acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    features = torch.cat(features)
    targets = torch.cat(targets)
    wc_intra, w_inter, c_inter = calculate_metrics(model.module.head.get_weight().cpu().T, features, targets)
    print(f"WC-Intra: {wc_intra}, W-Inter: {w_inter}, C-Inter: {c_inter}")
    if wandb.run:
        wandb.log({'wc_intra': wc_intra, 'w_inter': w_inter, 'c_inter': c_inter})
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,lr_scheduler, max_norm: float = 0,
                     mixup_fn: 'Mixup' = None, accum_iter=1,
                     model_ema=None
                    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = max(len(data_loader)//20,20)
    
    for itr,(samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        labels = targets.clone()

        lr_scheduler.step(epoch+itr/len(data_loader))
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # with torch.cuda.amp.autocast(enabled=True if loss_scaler is not None else False):
        with torch.amp.autocast('cuda',):
            outputs = model(samples, labels)
            loss = criterion( outputs, targets)
        
        loss /= accum_iter
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                        need_update=(itr + 1) % accum_iter == 0)
        else:
            loss.backward()
            if (itr + 1) % accum_iter == 0:
                optimizer.step()
        if (itr + 1) % accum_iter == 0:
            optimizer.zero_grad()
            
        torch.cuda.synchronize()
        # log metrics
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

            error_dict = {
                'model':model.state_dict(),
                'samples': samples,
                'targets':targets,
            }
            if wandb.run:
                
                torch.save(error_dict,wandb.run.dir+'error.pth')
            sys.exit(1)
       
        lr = optimizer.param_groups[-1]["lr"]
        if wandb.run: 
            log = {'loss':loss, 'lr':lr}
            wandb.log(log)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def pre_train(args,backbone,data_loader_train, data_loader_val):
    
    
    if isinstance(backbone, timm.models.vision_transformer.VisionTransformer):
        in_features = backbone.num_features
        backbone.head = nn.Identity()
    else:
        in_features = backbone.fc.weight.shape[1]
        backbone.fc = nn.Identity() # remove the fc layer
    
    
    if args.pretrained_weights:
        load_pretrained_weights(backbone, args.pretrained_weights, checkpoint_key=args.checkpoint_key, prefix=args.prefix)
    head = MarginHead(in_features, args.nb_classes, margin_loss=args.margin_loss, 
                      embed_dim=args.embed_dim)
    model = MarginModel(backbone, head)
    print("Head: ", head)
    train(args, model,data_loader_train, data_loader_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])   
    parser.add_argument('--margin_loss', type=str, default='softmax', choices=['arcgrad', 'arcface', 'softmax'])
    parser.add_argument('--embed_dim', type=int, default=512)
    args = parser.parse_args()
    # hack to add pre_train to before train
    eval_cls.train_one_epoch = train_one_epoch
    eval_cls.evaluate = evaluate
    eval_cls.train = pre_train
    eval_cls.main(args)
