#!/usr/bin/env python
"""
Implementation of Supcon: Supervised Contrastive Learning

"""
import argparse
import torch
import torch.distributed as dist
import math
import sys
import wandb

from typing import Iterable, Optional
from timm.data import Mixup
import torch.nn.functional as F
from torch import nn

from vitookit.models.build_model import build_head
from vitookit.utils import misc
from vitookit.utils.helper import *

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

@gin.configurable
class SupCon(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 embed_dim=512,
                 out_dim=128,
                 mlp_dim=2048, 
                 type='identical',
                 temperature=0.1,
                 sup_loss=False, # whether to use supervised loss
                 num_classes=1000):
        super(SupCon, self).__init__()
        assert type in ['arbitrary','identical','distinct']
        self.type = type
        self.s = 1 / temperature
        # self.s = nn.Parameter(torch.tensor(1 / temperature, requires_grad=True))
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.sup_loss = sup_loss
        
        self.embed_dim = embed_dim
        self.backbone = backbone
        self.projector = nn.Linear(embed_dim,out_dim)
        
        
        self.cls_head = nn.Linear(embed_dim,num_classes)
        self.views = 2
        # self.loss_fn = SupConLoss(temperature=temperature)
        

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
        loss_sup = F.cross_entropy(predict, targets)
        z = self.projector(rep)

        y1 = targets.clone()
        if self.type == 'identical':
            y2 = targets
        elif self.type == 'distinct':
            y2 = (targets + torch.randint(1,self.num_classes,(len(y1),),device=targets.device))%self.num_classes
        else:
            y2 = targets[torch.randperm(len(y1),device=targets.device)].contiguous()

        loss_disparate = self.disparate_loss(z,z,y1,y2)
        loss = loss_sup + loss_disparate
        self.log['loss_sup'] = loss_sup.item()
        self.log['loss_disparate'] = loss_disparate.item()
        self.log['z@std'] = z.std(0).mean().item()
        self.log['z@norm'] = z.norm(2,dim=-1).mean().item()
        self.log['rep@norm'] = rep.norm(2,dim=-1).mean().item()
        return loss, self.log

    def forward(self, samples, **kwargs):
        rep = self.backbone(samples)
        predict = self.cls_head(rep.detach())
        return predict
    
    
    def disparate_loss(self, z1, k2, y1, posy):
        k2 = misc.concat_all_gather_grad(k2)
        
        fz1,fz2 = F.normalize(z1,p=2,dim=-1), F.normalize(k2,p=2,dim=-1)
        cosine = fz1 @ fz2.t()  # b x N
        logits = cosine * self.s

        all_y1 = misc.concat_all_gather(y1)
        c1_mask = (y1.unsqueeze(1) == all_y1.unsqueeze(0)) # exclude samples from y1
        c2_mask = (posy.unsqueeze(1) == all_y1.unsqueeze(0)) # exclude samples from y2
        class_mask = c1_mask | c2_mask
        neg_mask = ~class_mask

        loss = multipos_ce_loss(logits, c2_mask, neg_mask)
        
        return loss

from vitookit.evaluation import eval_cls
_train = eval_cls.train
def pre_train(args,model,data_loader_train, data_loader_val):
    if hasattr(model,'fc'):
        embed_dim = model.fc.in_features
        model.fc = torch.nn.Identity()
    else:
        embed_dim = model.embed_dim
        model.head = torch.nn.Identity()
    if wandb.run:
        try:
            wandb.watch(model.layer4[2],log='all')
        except:
            pass
    model = SupCon(model,num_classes=args.nb_classes, embed_dim=embed_dim)
    model.views = args.ra
    model.cuda()
    _train(args,model,data_loader_train, data_loader_val)


def train_one_epoch(model: torch.nn.Module, _criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,lr_scheduler, max_norm: float = 0,
                     mixup_fn: Optional[Mixup] = None, accum_iter=1,
                    model_ema: Optional[torch.nn.Module] = None
                    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = max(len(data_loader)//20,20)
    criterion = model.module.criterion if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.criterion
    
    for itr,(samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        lr_scheduler.step(epoch+itr/len(data_loader))
        # if mixup_fn is not None: disable mixup for SupCon
        #     samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast('cuda',enabled=True if loss_scaler is not None else False):
            loss,log = criterion(samples, targets)
        
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
                torch.save(error_dict, args.output_dir+'/error.pth')
            dist.destroy_process_group()
            sys.exit(1)
        # this attribute is added by timm on one optimizer (adahessian)
        # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if model_ema is not None:
            model_ema.update(model)
        lr = optimizer.param_groups[-1]["lr"]
        if wandb.run: 
            log.update({'loss':loss_value, 'lr':lr})
            wandb.log(log)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SupCon training script', parents=[eval_cls.get_args_parser()])
    args = parser.parse_args()
    # hack to replace the train function
    args.mixup=0
    args.cutmix=0
    
    eval_cls.train = pre_train
    eval_cls.train_one_epoch = train_one_epoch
    eval_cls.main(args)
