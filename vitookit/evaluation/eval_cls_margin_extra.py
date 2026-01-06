#!/usr/bin/env python
"""
Supervised Learning with Regularized Margin Loss. Sweep lambda from 0.1,1,10.

export WANDB_TAGS=regularization
WANDB_NAME=sl-vit_tiny-0 sbatch --gpus=1 ~/storchrun.sh vitookit/evaluation/eval_cls_margin_extra.py --data_location $DATA_PATH --data_set CIFAR100  --model vit_tiny_patch16_224 --input_size 32 --gin build_model.patch_size=2 build_model.img_size=32   --batch_size 512 --epochs 1000 --warmup_epochs 20 --output_dir ~/storage/experiments/regularization/sl-vit_tiny-0

WANDB_NAME=sl-vit_tiny-0.1 sbatch --gpus=1 ~/storchrun.sh vitookit/evaluation/eval_cls_margin_extra.py --data_location $DATA_PATH --data_set CIFAR100  --model vit_tiny_patch16_224 --input_size 32 --gin build_model.patch_size=2 build_model.img_size=32   --batch_size 512 --epochs 1000 --warmup_epochs 20 --lam 0.1 --output_dir ~/storage/experiments/regularization/sl-vit_tiny-0.1

WANDB_NAME=sl-vit_tiny-1 sbatch --gpus=1 ~/storchrun.sh vitookit/evaluation/eval_cls_margin_extra.py --data_location $DATA_PATH --data_set CIFAR100  --model vit_tiny_patch16_224 --input_size 32 --gin build_model.patch_size=2 build_model.img_size=32   --batch_size 512 --epochs 1000 --warmup_epochs 20 --lam 1 --output_dir ~/storage/experiments/regularization/sl-vit_tiny-1

WANDB_NAME=sl-vit_tiny-10 sbatch --gpus=1 ~/storchrun.sh vitookit/evaluation/eval_cls_margin_extra.py --data_location $DATA_PATH --data_set CIFAR100  --model vit_tiny_patch16_224 --input_size 32 --gin build_model.patch_size=2 build_model.img_size=32   --batch_size 512 --epochs 1000 --warmup_epochs 20 --lam 10 --output_dir ~/storage/experiments/regularization/sl-vit_tiny-10

Baseline supcon and simlap:
WANDB_NAME=supcon-vit_tiny sbatch --gpus=1 ~/storchrun.sh vitookit/evaluation/supcon.py --data_location $DATA_PATH --data_set CIFAR100  --model vit_tiny_patch16_224 --input_size 32 --gin build_model.patch_size=2 build_model.img_size=32   --batch_size 512 --epochs 1000 --warmup_epochs 20 --output_dir ~/storage/experiments/regularization/supcon-vit_tiny

WANDB_NAME=simlap-vit_tiny sbatch --gpus=1 ~/storchrun.sh vitookit/evaluation/simlap.py --data_location $DATA_PATH --data_set CIFAR100  --model vit_tiny_patch16_224 --input_size 32 --gin build_model.patch_size=2 build_model.img_size=32   --batch_size 512 --epochs 1000 --warmup_epochs 20 --output_dir ~/storage/experiments/regularization/simlap-vit_tiny

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


from vitookit.models.losses import Softmax, ArcGrad, ArcFace
from vitookit.utils.helper import post_args, load_pretrained_weights
from vitookit.utils import misc
from vitookit.models.build_model import build_model
import wandb

from vitookit.evaluation.eval_cls import get_args_parser,train
from vitookit.evaluation import eval_cls
from timm.layers import (convert_splitbn_model, convert_sync_batchnorm,
                         set_fast_norm)

LAM_REG = 0.0

class MarginHead(nn.Module):
    r"""Implement of classification head with margin loss:
        Args:
            in_features: size of each input sample
            num_classes: number of classes
        """
    def __init__(self, in_features, num_classes, 
                 s = 10,
                 margin_loss='softmax',
                 embed_dim=512):
        super(MarginHead, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, embed_dim*4, bias=False),
            nn.ReLU(), nn.BatchNorm1d(embed_dim*4),
            nn.Linear(embed_dim*4, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, affine=False),
        )
        self.weight = nn.utils.weight_norm(
            nn.Linear(embed_dim, num_classes, bias=False)
        )
        self.weight.weight_g.requires_grad = False # freeze the magnitude of the weight        
        self.weight.weight_g.fill_(1)
        
        if margin_loss == 'arcgrad':
            self.margin = ArcGrad(num_classes, s)
        elif margin_loss == 'arcface':
            self.margin = ArcFace(num_classes, s)
        elif margin_loss == 'softmax':
            self.margin = Softmax(num_classes, s)
        else:
            raise ValueError(f'Invalid margin loss: {margin_loss}')
        
        
    def get_weight(self):
        """Get the weight of the classifier: num_classes x embed_dim"""
        return self.weight.weight_v.detach()

    def forward(self, input, labels=None,return_z=False, return_reg=False):
        z = self.fc(input)
        z = F.normalize(z, dim=1)
        cosine = self.weight(z)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        reg = None
        if return_reg:
            if labels is None:
                raise ValueError("labels must be provided when return_reg=True")
            # Compute sum of elementwise products between z and all non-target class vectors
            # weight vectors are normalized by weight norm (weight property)
            weight_vectors = self.weight.weight_v  # (num_classes, embed_dim)
            # B x C x D elementwise products then sum over D
            elem_prod = (z.unsqueeze(1) * weight_vectors.unsqueeze(0)).abs().sum(dim=2)
            mask = torch.ones_like(elem_prod, dtype=torch.bool)
            mask.scatter_(1, labels.view(-1, 1), False)  # zero out target class contributions
            reg = (elem_prod * mask.float()).mean()
        
        if self.training:
            logits = self.margin(cosine,labels)
            
        else:
            logits = cosine
        if return_z and return_reg:
            return logits, z, reg
        elif return_z:
            return logits, z
        elif return_reg:
            return logits, reg
        return logits
        
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
            outputs, reg = model(samples, labels, return_reg=True)
            loss_cls = criterion(outputs, targets)
            loss = loss_cls - LAM_REG * reg
            
        
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
            log = {'loss':loss_value, 'lr':lr, 'loss_cls': loss_cls.item(), 'reg': reg.item()}
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
    parser.add_argument('--lam', type=float, default=0)
    args = parser.parse_args()
    # hack to add pre_train to before train
    LAM_REG = args.lam
    eval_cls.train_one_epoch = train_one_epoch
    eval_cls.evaluate = evaluate
    eval_cls.train = pre_train
    eval_cls.main(args)
