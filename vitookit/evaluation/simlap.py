#!/usr/bin/env python
"""
@article{wu2024rethinking,
  title={Rethinking positive pairs in contrastive learning},
  author={Wu, Jiantao and Atito, Sara and Feng, Zhenhua and Mo, Shentong and Kitler, Josef and Awais, Muhammad},
  journal={arXiv preprint arXiv:2410.18200},
  year={2024}
}
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

def apply_gate(gate, x1, x2):
    x1 = torch.einsum("bk,bk->bk",x1,gate)
    x2 = torch.einsum("nk,bk->bnk",x2,gate)
    x1 =  F.normalize(x1,p=2,dim=-1)
    x2 =  F.normalize(x2,p=2,dim=-1)
    return x1, x2

def contrast(x1,x2):
    if x2.dim() == 3:
        logits =  torch.einsum("bj,bnj->bn",x1,x2)
    else:
        logits = x1 @ x2.t()
    return logits

@gin.configurable
class SimLAP(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 embed_dim=512,
                 out_dim=256,
                 mlp_dim=2048, 
                 type='arbitrary',
                 temperature=0.1,
                 sup_loss=False, # whether to use supervised loss
                 num_classes=1000):
        super(SimLAP, self).__init__()
        assert type in ['arbitrary','identical','distinct']
        self.type = type
        self.s = 1 / temperature
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.sup_loss = sup_loss
        
        self.embed_dim = embed_dim
        self.backbone = backbone
        self.projector = build_head(2,embed_dim,mlp_dim,out_dim, last_norm='ln')
        self.filter = Filter(num_classes=num_classes,embed_dim=out_dim)
        
        self.cls_head = nn.Linear(embed_dim,num_classes)
        

    @torch.jit.ignore
    def no_weight_decay(self) -> "Set[str]":
        """Set of parameters that should not use weight decay."""
        if hasattr(self.backbone, 'no_weight_decay'):
            backbone = self.backbone.no_weight_decay()
            return {'backbone.'+k for k in backbone}
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> "Dict[str, Union[str, List]]":
        """Create regex patterns for parameter grouping.

        Args:
            coarse: Use coarse grouping.

        Returns:
            Dictionary mapping group names to regex patterns.
        """
        
        return dict(
            stem=r'cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'backbone\.blocks\.(\d+)', None), (r'backbone\.norm', (99999,))],
            head=r'projector|cls_head|filter'
        )

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
        k2 = misc.concat_all_gather_grad(k2)
        gate = self.filter.gate(y1,posy)
        fz1, fz2 = apply_gate(gate, z1, k2)
        # fz1,fz2 = self.filter(z1, k2, y1,posy)
        cosine = contrast(fz1,fz2)
        scale = self.s
        self.log['scale'] = scale
        # logits = contrast(z1*gate*gate,k2)
        
        logits = scale * cosine
        
        c1_mask = (y1.unsqueeze(1) == (y1).unsqueeze(0)) # exclude samples from y1
        c2_mask = (posy.unsqueeze(1) == (y1).unsqueeze(0)) # exclude samples from y2
        class_mask = c1_mask|c2_mask
        neg_mask = ~class_mask

        loss = multipos_ce_loss(logits, c2_mask, neg_mask)
        
        # self.log['cosine@c1'] = ((cosine*c1_mask).sum()/c1_mask.sum()).item()
        # self.log['cosine@c2'] = ((cosine*c2_mask).sum()/c2_mask.sum()).item()
        # self.log['cosine@neg'] = ((cosine*neg_mask).sum()/neg_mask.sum()).item()
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
    
    model = SimLAP(model,num_classes=args.nb_classes, embed_dim=embed_dim)
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
        # if mixup_fn is not None: disable mixup for simlap
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
    parser = argparse.ArgumentParser('Simlap training script', parents=[eval_cls.get_args_parser()])
    args = parser.parse_args()
    # hack to replace the train function
    eval_cls.train = pre_train
    eval_cls.train_one_epoch = train_one_epoch
    eval_cls.main(args)
