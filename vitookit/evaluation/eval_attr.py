#!/usr/bin/env python
"""
Example:
vitrun  --nproc_per_node=8 eval_attr.py --data_location ~/data/ --data_set rival10 --fraction 0.1  --output_dir outputs/semi

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
from vitookit.evaluation.eval_cls import get_args_parser, NativeScaler, convert_sync_batchnorm, create_optimizer, create_scheduler, train_one_epoch,  restart_from_checkpoint, log_metrics, os, sys, json, Path
import wandb


from vitookit.datasets import rival10
def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (x-mean)/std
class RIVAL10AttributeDataset(rival10.LocalRIVAL10):
    def __getitem__(self, i):
        out = super().__getitem__(i)
        return normalize(out['img']), out['attr_labels']

def multilabel_loss(output, target):
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(output, target.float())
    
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
            wandb.init(job_type='attr',dir=args.output_dir,resume=True, 
                   config=args.__dict__)
        except:
            pass
    
    if args.data_set == 'rival10':
        dataset_train = RIVAL10AttributeDataset(args.data_location, train=True)
        dataset_val = RIVAL10AttributeDataset(args.data_location, train=False)
        args.nb_classes = 18 # number of attributes
    else:
        raise NotImplementedError
        
    print("Load dataset:", dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
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
    
    model = build_model(num_classes=args.nb_classes)

    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key=args.checkpoint_key, prefix=args.prefix)
    model.requires_grad_(False)
    if hasattr(model,'fc'):
        model.fc.requires_grad_(True)
    else:
        model.head.requires_grad_(True)
    print(f"Built Model ", model)
    train(args, model,data_loader_train, data_loader_val)


def train(args,model,data_loader_train, data_loader_val):
    """Use BCE loss to train the model
    """
    if args.compile:
        model = torch.compile(model)   
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True 
    model = convert_sync_batchnorm(model)
    import torch
    device = torch.device(args.device)
    
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

    
    criterion = multilabel_loss

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
            args.clip_grad, accum_iter=args.accum_iter
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
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, output_dir / checkpoint_path)
            else:
                misc.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
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

@torch.no_grad()
def evaluate(data_loader, model, device,return_preds=False ):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    preds = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True, dtype=torch.long)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = multilabel_loss(output, target)
        preds.append(output.argmax(1).cpu())

        pred = (output>0).long().flatten()
        target = target.flatten()
        # we use precision and f1-score to measure the performance
        import torcheval.metrics.functional as  metrics
        pr = metrics.binary_precision(pred,target)
        recall = metrics.binary_recall(pred,target)
        auroc = metrics.binary_auroc(pred,target)
        f1 = metrics.binary_f1_score(pred,target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['pr'].update(pr.item(), n=batch_size)
        metric_logger.meters['recall'].update(recall.item(), n=batch_size)
        metric_logger.meters['auroc'].update(auroc.item(), n=batch_size)
        metric_logger.meters['f1'].update(f1.item(), n=batch_size)
        acc1 = (pred==target).float().mean()
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes() 
    preds = torch.cat(preds)
    if return_preds:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()},preds
    else:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of the dataset to use")
    args = parser.parse_args()
    main(args)
