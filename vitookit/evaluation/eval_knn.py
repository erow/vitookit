# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
from pathlib import Path
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms
from vitookit.datasets import build_dataset

from vitookit.utils.helper import load_pretrained_weights, post_args
from vitookit.models.build_model import build_model
from vitookit.utils import misc

def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDatasetWrap( build_dataset(args,True,transform)[0])
    dataset_val = ReturnIndexDatasetWrap(build_dataset(args,False,transform)[0])
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
       
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    
    model = build_model(num_classes=0)
    if args.pretrained_weights:
        load_pretrained_weights(model, args.pretrained_weights, checkpoint_key=args.checkpoint_key, prefix=args.prefix)
        
    
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda().requires_grad_(False)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features,train_labels = extract_features(model, data_loader_train, True)
    print("Extracting features for val set...")
    test_features,test_labels = extract_features(model, data_loader_val,True)
    
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = misc.MetricLogger(delimiter="  ")
    features = None
    labels = None
    for samples, labels_c, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        labels_c = labels_c.cuda(non_blocking=True)
        if multiscale:
            feats = misc.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            labels = torch.zeros(len(data_loader.dataset)).long()
            if use_cuda:
                features = features.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()
        
        labels_all = torch.empty(
            dist.get_world_size(),
            labels_c.size(0),
            dtype=labels_c.dtype,
            device=labels_c.device,
        )
        output_labels_l = list(labels_all.unbind(0))
        output_labels_all_reduce = torch.distributed.all_gather(output_labels_l, labels_c, async_op=True)
        output_labels_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
                labels.index_copy_(0, index_all, torch.cat(output_labels_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
                labels.index_copy_(0, index_all.cpu(), torch.cat(output_labels_l).cpu())
    return features,labels

import tqdm
@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000,dis_fn='euclidean'):
    top1, top5, total = 0.0, 0.0, 0
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in tqdm.tqdm(range(0, num_test_images, imgs_per_chunk)):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        if dis_fn=='euclidean':
            similarity = (features.unsqueeze(1) - train_features.unsqueeze(0)).square().mean(-1)
            distances, indices = similarity.topk(k, largest=False, sorted=True)
        elif dis_fn=='cosine':
            features=nn.functional.normalize(features, dim=1, p=2)
            train_features=nn.functional.normalize(train_features, dim=1, p=2)
            # print("train_features: ", train_features.shape)
            similarity = torch.mm(features, train_features.T)
            distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


class ReturnIndexDatasetWrap():
    def __init__(self,data) -> None:
        self.data=data
    
    def get_labels(self):
        for i in range(len(self.data)):
            yield self.data[i][1]
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, lab = self.data[idx]
        return img, lab, idx

def get_parser():
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[1, 10,20,50,100], nargs='+', type=int,
        help='Number of NN to use. 10 is usually working the best for small datasets and 20 for large datasets.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('-w', '--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--prefix", default=None, type=str, help="prefix of the model name")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default=None, type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_location', default='./data', type=str)
    parser.add_argument('--data_set', default='Pets', type=str)
    parser.add_argument('--output_dir',default='',type=str,help='path where to save, empty for no saving')
    parser.add_argument('--dis_fn',default='cosine', type=str, 
                        choices=['cosine','euclidean'])
    parser.add_argument('--use_cuda', default=False, action='store_true',)
    parser.add_argument('--bn',default=False,action='store_true', help="Apply batch normalization after extracting features. This is neccessary for MAE.")
    # configure
    parser.add_argument('--cfgs', nargs='+', default=[],
                        help='<Required> Config files *.gin.', required=False)
    parser.add_argument('--gin', nargs='+', 
                        help='Overrides config values. e.g. --gin "section.option=value"')

    
    return parser
if __name__ == '__main__':
    
    
    parser = get_parser()
    
    args = parser.parse_args()
    post_args(args)

    misc.init_distributed_mode(args)
    print("git:\n  {}\n".format(misc.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    
    if misc.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()
        if args.bn:
            mu = train_features.mean(dim=0, keepdim=True)
            std = train_features.std(dim=0, keepdim=True)
            
            train_features = (train_features-mu)/(std+1e-6)
            test_features = (test_features-mu)/(std+1e-6)
        
        print("Features are ready!\nStart the k-NN classification.")
        if args.output_dir:
            output_dir=Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None
        

        for k in args.nb_knn:
            top1, top5 = knn_classifier(train_features, train_labels,
                test_features, test_labels, k, args.temperature,dis_fn=args.dis_fn)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
            
            log_stats = {f'{k}/acc1':top1,f'{k}/acc5':top5}
            

            if output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")            
    dist.barrier()
