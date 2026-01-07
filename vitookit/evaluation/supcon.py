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
import numpy as np

from typing import Iterable, Optional
from timm.data import Mixup
import torch.nn.functional as F
from torch import nn

from vitookit.models.build_model import build_head
from vitookit.utils import misc
from vitookit.utils.helper import *

class PKSampler(torch.utils.data.Sampler):
    """
    PK Sampler for Supervised Contrastive Learning.
    Samples P classes with K samples each per batch, ensuring each sample
    has at least K-1 positives in the batch.
    
    Supports distributed training by partitioning classes across ranks.
    """
    
    def __init__(self, labels, p_classes, k_samples, num_replicas=None, rank=None, seed=0):
        """
        Args:
            labels: List or array of class labels for each sample
            p_classes: Number of classes to sample per batch
            k_samples: Number of samples per class
            num_replicas: Number of distributed processes (default: world_size)
            rank: Rank of current process (default: current rank)
            seed: Random seed for reproducibility
        """
        self.labels = np.array(labels)
        self.p = p_classes
        self.k = k_samples
        self.seed = seed
        self.epoch = 0
        
        # Distributed settings
        if num_replicas is None:
            num_replicas = misc.get_world_size()
        if rank is None:
            rank = misc.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        
        # Build class to indices mapping
        self.classes = np.unique(self.labels)
        self.class_to_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        
        # Calculate number of batches per epoch
        self.batch_size = self.p * self.k
        self.num_batches = len(self.labels) // (self.batch_size * self.num_replicas)
        self.total_size = self.num_batches * self.batch_size
        
    def __iter__(self):
        # Deterministic shuffling based on epoch and seed
        rng = np.random.RandomState(self.seed + self.epoch)
        
        for _ in range(self.num_batches):
            # Sample P classes (each rank samples independently with different random state)
            batch_rng = np.random.RandomState(rng.randint(0, 2**31) + self.rank)
            selected_classes = batch_rng.choice(self.classes, self.p, replace=False)
            
            indices = []
            for c in selected_classes:
                class_indices = self.class_to_indices[c]
                # Sample with replacement if class has fewer than K samples
                replace = len(class_indices) < self.k
                sampled = batch_rng.choice(class_indices, self.k, replace=replace)
                indices.extend(sampled.tolist())
            
            yield from indices
    
    def __len__(self):
        return self.total_size
    
    def set_epoch(self, epoch):
        """Set epoch for deterministic shuffling across epochs."""
        self.epoch = epoch

def multipos_ce_loss(logits, pos_mask,neg_mask=None):
    if neg_mask is None:
        neg_mask = ~pos_mask
    # detach the mean to avoid gradient flow
    # logits = logits - logits.mean(1,keepdim=True).detach() 
    similarity = logits.exp()
    N = similarity.size(0)
 
    # InfoNCE loss 
    ## exclude the positives and class pairs
    neg = (similarity*neg_mask).sum(1,keepdim=True)
    loss = torch.sum(pos_mask* (torch.log(similarity + neg) - logits),dim=1)/pos_mask.sum(dim=1)
    loss = loss.mean()
   
    return loss

@gin.configurable
class SupCon(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 embed_dim=512,
                 out_dim=128,
                 mlp_dim=2048, 
                 temperature=0.1,
                 sup_loss=False, # whether to use supervised loss
                 k_samples=2,
                 num_classes=1000):
        super(SupCon, self).__init__()
        
        self.temperature = temperature
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.sup_loss = sup_loss
        self.k_samples = k_samples
        
        self.embed_dim = embed_dim
        self.backbone = backbone
        self.projector = nn.Linear(embed_dim,out_dim)
        
        self.cls_head = nn.Linear(embed_dim,num_classes)
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
        zs = self.projector(rep)
        zs = zs.reshape(-1, self.k_samples, self.out_dim)  # [P, K, D]
        y = targets[::self.k_samples].contiguous()  # One label per class [P]
        
        # Contrast each sample with every other sample of the same class
        loss_contrastive = 0
        
        for i in range(self.k_samples):
            for j in range(i+1,self.k_samples):
                # loss_contrastive += NT_Xent(zs[:, i], zs[:, j], self.temperature)
                loss_contrastive += self.contrastive_loss(zs[:, i], zs[:, j], y)
        loss_contrastive /= self.k_samples * (self.k_samples - 1)

        loss = loss_sup + loss_contrastive
        self.log['loss_sup'] = loss_sup.item()
        self.log['loss_contrastive'] = loss_contrastive.item()
        self.log['z@std'] = zs.std(0).mean().item()
        self.log['z@norm'] = zs.norm(2, dim=-1).mean().item()
        self.log['rep@norm'] = rep.norm(2, dim=-1).mean().item()
        return loss, self.log

    def forward(self, samples, **kwargs):
        rep = self.backbone(samples)
        predict = self.cls_head(rep.detach())
        return predict
    
    
    def contrastive_loss(self, z1, z2, y):
        """
        Paired contrastive loss: one sample per class at a time.
        z1[i] and z2[i] are from the same class -> positive pair.
        All samples from different classes -> negatives.
        """
        z2_gathered = misc.concat_all_gather_grad(z2.contiguous())
        y_gathered = misc.concat_all_gather(y)
        
        fz1 = F.normalize(z1, p=2, dim=-1)
        fz2 = F.normalize(z2_gathered, p=2, dim=-1)
        
        # Similarity: [P_local, P_global]
        logits = (fz1 @ fz2.t()) / self.temperature
        
        batch_size = z1.size(0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Positive: z1[i] pairs with z2_gathered[rank*B + i] (one-to-one)
        pos_idx = rank * batch_size + torch.arange(batch_size, device=z1.device)
        pos_logits = logits[torch.arange(batch_size, device=z1.device), pos_idx]  # [P]
        
        # Negative mask: different class samples
        neg_mask = (y.unsqueeze(1) != y_gathered.unsqueeze(0))
        neg_exp_sum = (logits.exp() * neg_mask).sum(dim=1)  # [P]
        
        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        loss = -pos_logits + torch.log(pos_logits.exp() + neg_exp_sum)
        loss = loss.mean()
        
        return loss

from vitookit.evaluation import eval_cls
from vitookit.datasets.build_dataset import build_dataset, build_transform

def build_pk_loader(args):
    """Build data loaders with PK sampling for supervised contrastive learning."""
    transform = build_transform(is_train=True, args=args)
    dataset_train, nb_classes = build_dataset(args=args, is_train=True, trnsfrm=transform)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    if nb_classes:
        args.nb_classes = nb_classes
    
    print(f"Building PK sampler: P={args.pk_classes} classes, K={args.pk_samples} samples/class")
    print(f"Effective batch size per GPU: {args.pk_classes * args.pk_samples}")
    
    # Extract labels from dataset
    if hasattr(dataset_train, 'targets'):
        labels = dataset_train.targets
    elif hasattr(dataset_train, 'samples'):
        labels = [s[1] for s in dataset_train.samples]
    else:
        raise ValueError("Dataset must have 'targets' or 'samples' attribute for PK sampling")
    
    # Create PK sampler
    sampler_train = PKSampler(
        labels=labels,
        p_classes=args.pk_classes,
        k_samples=args.pk_samples,
        seed=args.seed
    )
    
    # Validation sampler (standard sequential)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    # Override batch_size to match PK batch
    pk_batch_size = args.pk_classes * args.pk_samples
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=pk_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * pk_batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True,
    )
    
    return data_loader_train, data_loader_val

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
    model = SupCon(model,num_classes=args.nb_classes, embed_dim=embed_dim, k_samples=args.pk_samples)
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
        loss_value = log['loss_contrastive']

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
            log.update({'loss':loss, 'lr':lr})
            wandb.log(log)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def supcon_main(args):
    """Main function for SupCon training with optional PK sampling."""
    misc.init_distributed_mode(args)
    misc.fix_random_seeds(args.seed)
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    
    print(args)
    post_args(args)
    
    # Logging
    if args.output_dir and misc.is_main_process():
        try:
            wandb.init(job_type='supcon', dir=args.output_dir, resume=not args.resume is None,
                       config=args.__dict__)
        except:
            pass
    
    # Use PK sampling if enabled, otherwise fall back to standard loader
    if args.pk_classes > 0 and args.pk_samples > 0:
        data_loader_train, data_loader_val = build_pk_loader(args)
    else:
        from vitookit.datasets.build_dataset import build_loader
        data_loader_train, data_loader_val = build_loader(args)
    
    # build the backbone
    from vitookit.models.build_model import build_model
    model = build_model(args.model, num_classes=args.nb_classes)
    
    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights, 
                                checkpoint_key=args.checkpoint_key, prefix=args.prefix)
    
    pre_train(args, model, data_loader_train, data_loader_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SupCon training script', parents=[eval_cls.get_args_parser()])
    
    # PK sampling arguments
    parser.add_argument('--pk_classes', type=int, default=0,
                        help='Number of classes per batch for PK sampling (P). Set >0 to enable.')
    parser.add_argument('--pk_samples', type=int, default=2,
                        help='Number of samples per class for PK sampling (K). Default: 2')
    
    args = parser.parse_args()
    
    # Disable mixup/cutmix for SupCon (not compatible with contrastive learning)
    args.mixup = 0
    args.cutmix = 0
    
    # If PK sampling is enabled, update batch_size to match
    if args.pk_classes > 0 and args.pk_samples > 0:
        args.batch_size = args.pk_classes * args.pk_samples // misc.get_world_size()
        print(f"PK Sampling enabled: P={args.pk_classes}, K={args.pk_samples}, batch_size={args.batch_size}")
    
    eval_cls.train_one_epoch = train_one_epoch
    supcon_main(args)
