"""Evaluate models with linear probing.

Usage: vitrun 
"""
from PIL import Image # hack to avoid `CXXABI_1.3.9' not found error
import argparse
import datetime
import json
import sys
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import wandb
from vitookit.models.build_model import build_model

from vitookit.utils import misc
from vitookit.datasets import build_dataset

from torchvision.transforms import *
from vitookit.utils.helper import post_args, load_pretrained_weights, log_metrics, restart_from_checkpoint
from timm.models.layers import trunc_normal_
from timm.layers import (convert_splitbn_model, convert_sync_batchnorm,
                         set_fast_norm)

from vitookit.utils.lars import LARS
import gin



def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--ckpt_freq', default=5, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument("--compile", action='store_true', default=False, help="compile model with PyTorch 2.0")
    parser.add_argument("--checkpoint_key", default=None, type=str, help="checkpoint key to load")
    parser.add_argument("--prefix", default=None, type=str, help="prefix of the model name")

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    


    # * Finetuning params
    parser.add_argument('-w', '--pretrained_weights', default='',
                        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_set',default='IN1K',type=str)
    parser.add_argument('--data_location', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=None, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # configure
    parser.add_argument('--cfgs', nargs='+', default=[],
                        help='<Required> Config files *.gin.', required=False)
    parser.add_argument('--gin', nargs='+', 
                        help='Overrides config values. e.g. --gin "section.option=value"')
    

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    post_args(args)
    import torch
    print("args: ", args)
    print("configure: ", gin.config_str())
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    train_transform = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = Compose([
        Resize(256, interpolation=3),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
        
    dataset_train,nb_classes = build_dataset(is_train=True, args=args,trnsfrm=train_transform)
    dataset_val,_ = build_dataset(is_train=False, args=args,trnsfrm=val_transform)
    if args.nb_classes is None: args.nb_classes=nb_classes
    
    print("Dataset = ", str(dataset_train))
    print("len(Dataset), nb_classes = ", len(dataset_train), args.nb_classes)
    
    if True:  
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.output_dir and not args.eval:
        wandb.init(job_type='linprob',
                   dir=args.output_dir,
                   config=args.__dict__,
                   resume=True)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    model = build_model(num_classes=args.nb_classes)
    
    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key=args.checkpoint_key, prefix=args.prefix)
    # for linear prob only
    model = convert_sync_batchnorm(model)
    # hack: revise model's head with BN
    model.eval()
    if hasattr(model, 'head'):
        trunc_normal_(model.head.weight, std=0.01)
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        learnable_module = model.head
        
    elif hasattr(model, 'fc'):
        trunc_normal_(model.fc.weight, std=0.01)
        # model.fc = torch.nn.Sequential(torch.nn.BatchNorm1d(model.fc.in_features, affine=False, eps=1e-6), model.fc)
        learnable_module = model.fc
    else:
        print("model head not found")
    
    # freeze all but the head
    learnable_module.requires_grad_(True)
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in learnable_module.named_parameters():
        p.requires_grad = True 
        
    if args.compile:
        model = torch.compile(model)    
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True 
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(learnable_module))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

   
    model_without_ddp = model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    optimizer = LARS(learnable_module.parameters(), 
                     lr=args.lr, momentum=0.9,
                     weight_decay=args.weight_decay) # LARS for large batch training
    
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))
    
    if args.resume:        
        run_variables={"args":dict(),"epoch":0}
        restart_from_checkpoint(args.resume,
                                optimizer=optimizer,
                                model=model_without_ddp,
                                scaler=loss_scaler,
                                run_variables=run_variables)
        # args = run_variables['args']
        args.start_epoch = run_variables["epoch"] + 1
        print("Resuming from epoch %d" % args.start_epoch)

    if args.eval:
        assert args.world_size == 1
        model.to(device).eval()
        with torch.no_grad():    
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

            preds = []
            targets = []
            for images, target in data_loader_val:
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True, dtype=torch.long)
                output = model(images)            
                preds.append(output.argmax(1).cpu())
                targets.append(target.cpu())
            
            preds = torch.cat(preds)
            targets = torch.cat(targets)

        from sklearn.metrics import classification_report
        report_dict = classification_report(targets.cpu().numpy(), preds.cpu().numpy(), output_dict=True)
        print(report_dict)
        if output_dir:
            with (output_dir / "report.txt").open("w") as f:
                f.write(json.dumps(report_dict) + "\n")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
    output_dir = Path(args.output_dir) if args.output_dir else None
        
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args
        )
        log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
                
        if epoch % args.ckpt_freq==0 or epoch == args.epochs-1:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            log_stats.update({f'test/{k}': v for k, v in test_stats.items()})

            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')
        else:
            test_stats={}

        if args.output_dir and misc.is_main_process():
            if epoch % args.ckpt_freq == 0 or epoch == args.epochs-1:
                ckpt_path = output_dir / 'checkpoint.pth'
                misc.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, ckpt_path) 
            
            if wandb.run:
                wandb.log(log_stats)
                
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
                
            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # basename = os.path.basename(__file__)
    # log_metrics(basename, log_stats, args)


from timm.utils import accuracy

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = None,
                    mixup_fn = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        if wandb.run:
            wandb.log({'train/loss':loss,
                       'opt/lr':max_lr})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

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

    
import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

if __name__ == '__main__':
    parser = get_args_parser()    
    args = parser.parse_args()
    main(args)
