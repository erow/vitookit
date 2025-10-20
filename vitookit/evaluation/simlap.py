#!/usr/bin/env python
"""
@article{wu2024rethinking,
  title={Rethinking positive pairs in contrastive learning},
  author={Wu, Jiantao and Atito, Sara and Feng, Zhenhua and Mo, Shentong and Kitler, Josef and Awais, Muhammad},
  journal={arXiv preprint arXiv:2410.18200},
  year={2024}
}
"""
from ffcv.loader import  Loader, OrderOption
from vitookit.datasets.ffcv_transform import ThreeAugmentPipeline, SimplePipeline

import argparse
import datetime
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
from vitookit.models.build_model import build_head, build_model, create_backbone

from vitookit.datasets.build_dataset import build_dataset, build_transform
from timm.layers import (F, convert_splitbn_model, convert_sync_batchnorm,
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

from torch import nn

def multipos_ce_loss(logits, pos_mask,neg_mask=None):
    if neg_mask is None:
        neg_mask = ~pos_mask
    logits = logits - logits.mean(1,keepdim=True)
    similarity = logits.exp()
    N = similarity.size(0)
 
    # InfoNCE loss 
    ## exclude the positives and class pairs
    neg = (similarity*neg_mask).sum(1,keepdim=True)
    loss = torch.sum(pos_mask* (torch.log(similarity + neg) - logits))/pos_mask.sum()
    loss = loss.mean()
   
    return loss

class OpenGate(nn.Module):
    def __init__(self, embed_dim,num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def forward(self,y1,y2=None,log=None):
        bs = y1.size(0)
        gate = torch.ones(bs,self.embed_dim,device=y1.device)
        return gate


class BasicGate(OpenGate):
    def __init__(self, embed_dim, num_classes=1000, in_dim=512, 
                 mlp_dim=1024,
                 lam = 0, fuse=True):
        super().__init__(embed_dim,num_classes)
        
        self.mlp = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(), nn.BatchNorm1d(mlp_dim),
            nn.Linear(mlp_dim, embed_dim),            
        )
        self.label_embedding = nn.Embedding(num_classes,in_dim)
        self.lam = lam
        self.fuse = fuse

    def statistics(self):
        labels = torch.arange(self.num_classes).cuda()
        label_embeds = self.label_embedding(labels)
        logits = self.mlp(label_embeds)
        gates = logits.sigmoid()
        activation = gates.sum(1).mean()
        entropy = torch.distributions.Bernoulli(gates).entropy().mean()
        return activation, entropy
    
    def forward(self,y1,y2=None,log=None):
        if self.fuse:      
            if y2 is None:
                label_embeds = self.label_embedding(y1)
            else:
                label_embeds = (self.label_embedding(y1) + self.label_embedding(y2))/2
            
            logits = self.mlp(label_embeds)
            gate = logits.sigmoid()
        else:
            if y2 is None:
                gate = self.mlp(self.label_embedding(y1)).sigmoid()
            else:
                gate1 = self.mlp(self.label_embedding(y1)).sigmoid()
                gate2 = self.mlp(self.label_embedding(y2)).sigmoid()
                gate = gate1 * gate2
        return gate
    
class Filter(nn.Module):
    def __init__(self,num_classes, embed_dim, gate_fn=BasicGate):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate = gate_fn(embed_dim,num_classes=num_classes)
    
    def forward(self, x1,x2,y1,y2=None):
        gate = self.gate(y1,y2)
        x1 = torch.einsum("bk,bk->bk",x1,gate)
        x2 = torch.einsum("nk,bk->bnk",x2,gate)
        x1 =  F.normalize(x1,p=2,dim=-1)
        x2 =  F.normalize(x2,p=2,dim=-1)
        return x1, x2
    
    def contrast(self,x1,x2):
        logits =  torch.einsum("bj,bnj->bn",x1,x2)
        return logits


@gin.configurable
class SimLAP(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 out_dim=256,
                 mlp_dim=2048, 
                 type='arbitrary',
                 temperature=0.1,
                 sup_loss=False,
                 num_classes=1000):
        super(SimLAP, self).__init__()
        embed_dim = backbone.embed_dim
        assert type in ['arbitrary','identical','distinct']
        self.type = type
        self.s = 1/temperature
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.sup_loss = sup_loss
        
        self.embed_dim = embed_dim
        self.backbone = backbone
        self.projector = build_head(2,embed_dim,mlp_dim,out_dim, last_norm='ln')
        self.filter = Filter(num_classes=num_classes,embed_dim=out_dim)
        
        self.cls_head = nn.Linear(embed_dim,num_classes)

    @torch.no_grad()
    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        latent = self.backbone(x)
        proj = self.projector(latent)
        rep = dict(latent=latent,z=proj)
        return rep
    
    def criterion(self, samples, targets, **kwargs):
        self.log = {}
        rep = self.backbone(samples)
        if self.sup_loss:
            predict = self.cls_head(rep)
        else:            
            predict = self.cls_head(rep.detach())

        z = self.projector(rep)

        y1 = targets
        if self.type == 'identical':
            y2 = targets
        elif self.type == 'distinct':
            y2 = (targets + torch.randint(1,self.num_classes,(len(targets),),device=targets.device))%self.num_classes
        else:
            y2 = targets[torch.randperm(len(targets),device=targets.device)]

        loss = self.disparate_loss(z,z,y1,y2)
        loss += F.cross_entropy(predict, targets)

        self.log['z@std'] = z.std(0).mean().item()
        return loss, self.log

    def forward(self, samples, **kwargs):
        rep = self.backbone(samples)
        predict = self.cls_head(rep.detach())
        return predict
    
    
    def disparate_loss(self, z1, k2, y1, posy):
        # k2 = concat_all_gather(k2)
        fz1,fz2 = self.filter(z1, k2, y1,posy)
        
        scale = self.s
        self.log['scale'] = scale
        cosine = self.filter.contrast(fz1,fz2)
        logits = scale * cosine
        
        c1_mask = (y1.unsqueeze(1) == (y1).unsqueeze(0)) # exclude samples from y1
        c2_mask = (posy.unsqueeze(1) == (y1).unsqueeze(0)) # exclude samples from y2
        class_mask = c1_mask|c2_mask
        neg_mask = ~class_mask

        loss = multipos_ce_loss(logits,c2_mask, neg_mask)
        
        self.log['cosine@c1'] = ((cosine*c1_mask).sum()/c1_mask.sum()).item()
        self.log['cosine@c2'] = ((cosine*c2_mask).sum()/c2_mask.sum()).item()
        self.log['cosine@neg'] = ((cosine*neg_mask).sum()/neg_mask.sum()).item()
        return loss
    
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
    parser.add_argument("--model", default='resnet50', type=str, help="model name")
    parser.add_argument("--compile", action='store_true', default=False, help="compile model with PyTorch 2.0")
    parser.add_argument("--prefix", default=None, type=str, help="prefix of the model name")
    
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('-w', '--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default=None, type=str, help='Key to use in the checkpoint (example: "teacher")')


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
    parser.add_argument('--no_amp', action='store_true', default=False, help='Disable AMP (automatic mixed precision)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: blr * batch_size / 256), see --blr)')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
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
    
    parser.add_argument('--ThreeAugment', action='store_true', default=False) #3augment
    parser.add_argument('--src',action='store_true', default=False, 
                        help="Use Simple Random Crop (SRC) or Random Resized Crop (RRC). Use SRC when there is less risk of overfitting, such as on ImageNet-21k.")
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--ra', type=int, default=0, metavar='N', help="Repeated Augmentations. Use 0 for no repeated augmentations.")    

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
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
    parser.add_argument('--data_location', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IN1K', 
                            type=str, help='ImageNet dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int, help='number of classes')

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

class BatchAugmentation:
    # https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf
    def __init__(self, sampler, repeat=1):
        self.sampler = sampler
        self.repeat = repeat

    def __len__(self):
        return len(self.sampler)*self.repeat

    def __iter__(self):
        for index in self.sampler:
            for _ in range(self.repeat):
                yield index

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,lr_scheduler, max_norm: float = 0,
                     mixup_fn: Optional[Mixup] = None, accum_iter=1
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
        
        with torch.amp.autocast('cuda',):
            
            loss,log = criterion(samples,targets)  
            
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
        # this attribute is added by timm on one optimizer (adahessian)
        # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        # if model_ema is not None:
        #     model_ema.update(model)
        lr = optimizer.param_groups[-1]["lr"]
        if wandb.run: 
            wandb.log({'loss':loss, 'lr':lr})
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device,return_preds=False ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    preds = []
    targets = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        target = target.to(device, non_blocking=True, dtype=torch.long)

        # compute output
        # with torch.cuda.amp.autocast():
        output = model(images)
        loss = criterion(output, target)
        preds.append(output.argmax(1).cpu())
        targets.append(target.cpu())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


        
def main(args):
    args.distributed = False
    # fix the seed for reproducibility
    misc.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print(args)
    post_args(args)
    
    # build dataset
    if args.ThreeAugment:
        transform = three_augmentation(args.input_size, args.color_jitter, args.src)
    else:
        transform = build_transform(is_train=True, args=args)
    
    dataset_train, args.nb_classes = build_dataset(args=args, is_train=True, trnsfrm=transform)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
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
    backbone = build_model(args.model, num_classes=0)
    model = SimLAP(backbone, num_classes=args.nb_classes)

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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    optimizer = create_optimizer(args, model_without_ddp)
    
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

    criterion = model_without_ddp.criterion

    print("criterion = %s" % str(criterion))
    

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir and misc.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(config=args,resume=args.resume is not None)
        
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

        train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,lr_scheduler,
            args.clip_grad,  mixup_fn, accum_iter=args.accum_iter
        )

        if ((epoch + 1)%args.ckpt_freq == 0 or epoch + 1 == args.epochs):
            train_stats = evaluate(data_loader_train, model, device)
            print(f"Accuracy of the network on the train images: {train_stats['acc1']:.1f}%")
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
            
            
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                        **{f'test/{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,}
            if output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                
                state = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'epoch': epoch,
                }
                torch.save(state, output_dir / 'checkpoint.pth')
                if wandb.run:
                    wandb.log(log_stats)
            
        if output_dir and epoch % (args.epochs//20) == 0:
            state = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': loss_scaler.state_dict(),
                'args': args,
                'epoch': epoch,
            }
            torch.save(state, output_dir / f'checkpoint-{epoch:05d}.pth')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if output_dir and misc.is_main_process():
        state = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': loss_scaler.state_dict(),
            'args': args,
            'epoch': epoch,
        }
        torch.save(state, output_dir / 'checkpoint.pth')
        print('Saved checkpoint to', output_dir / 'checkpoint.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
