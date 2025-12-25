import numpy as np
import gin

from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage,RandomHorizontalFlip, View, Convert
from ffcv.transforms.color_jitter import RandomColorJitter
from ffcv.transforms.solarization import RandomSolarization
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv import Loader
from ffcv.loader import OrderOption
from ffcv.traversal_order import Random

import torch
import torchvision.transforms.v2 as tfms
from torchvision.transforms import functional as F
from torch import nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    
@gin.configurable
def SimplePipeline(img_size=224,scale=(0.2,1), ratio=(3.0/4.0, 4.0/3.0),device='cuda'):
    device = torch.device(device)
    image_pipeline = [
            RandomResizedCropRGBImageDecoder((img_size, img_size), scale=scale,ratio=ratio,),
            RandomHorizontalFlip(),          
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),  
            ToTensor(),  ToTorchImage(),
            ]
    label_pipeline = [IntDecoder(), ToTensor(),ToDevice(device), View(-1)]
    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    } 
    return pipelines    

@gin.configurable
def ValPipeline(img_size=224,ratio= 224/256,device='cuda'):
    device = torch.device(device)
    image_pipeline = [
            CenterCropRGBImageDecoder((img_size, img_size), ratio),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
            ToTensor(),  ToTorchImage(),
            ToDevice(device),
            ]
    label_pipeline = [IntDecoder(), ToTensor(),ToDevice(device), View(-1)]
    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    } 
    return pipelines  


class ThreeAugmentation(nn.Module):
    """Apply single transformation randomly picked from a list. This transform does not support torchscript."""

    def __init__(self, ):
        super().__init__()
        self.guassian_blur = tfms.GaussianBlur(3,sigma=(0.1,2))
        self.solarize = tfms.RandomSolarize(0,1)
        self.grayscale = tfms.RandomGrayscale(p=1)

    def __call__(self, x):
        op_index = torch.randint(0,3,(len(x),))
        for i,op in enumerate([self.guassian_blur,
                               self.solarize,
                               self.grayscale]):
            tf_mask = op_index == i
            x[tf_mask] = op(x[tf_mask])
        return x

    def extra_repr(self) -> str:
        return "GaussianBlur, Solarize, Grayscale"
        
@gin.configurable
def ThreeAugmentPipeline(img_size=224,scale=(0.08,1), color_jitter=None,device='cuda'):
    """
    ThreeAugmentPipeline: https://github.com/facebookresearch/deit/blob/main/augment.py
    """
    if not color_jitter is None: assert color_jitter >= 0 and color_jitter <= 1
    device = torch.device(device)
    image_pipeline = (
        # first_tfl 
        [   RandomResizedCropRGBImageDecoder((img_size, img_size), scale=scale,),
            RandomHorizontalFlip(),]+
        # second_tfl
        (   [RandomColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter,hue=0, p=0.5)] if color_jitter else []) + 
        # final_tfl
        [
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
            ToTensor(), ToTorchImage(),
            ToDevice(device),
            ThreeAugmentation(),
        ]) 
        
    label_pipeline = [IntDecoder(), ToTensor(),ToDevice(device),View(-1)]
    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    } 
    return pipelines   

@gin.configurable
def ColorJitterPipeline(img_size=224,scale=(0.08, 1.0),device='cuda'):
    device = torch.device(device)
    image_pipeline = [
        RandomHorizontalFlip(),
        RandomColorJitter(0.8, 0.4, 0.4, 0.2, p=0.1),
        RandomSolarization(128,p=0.2),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ToTensor(), ToTorchImage(),
        ToDevice(device,non_blocking=True),
        tfms.RandomGrayscale(p=0.1),
        tfms.GaussianBlur(3, sigma=(0.1, 2)),
    ]
    label_pipeline = [IntDecoder(), ToTensor(),ToDevice(device),View(-1)]
    # Pipeline for each data field
    from ffcv.pipeline import PipelineSpec
    pipelines = {
        'image': PipelineSpec("image",RandomResizedCropRGBImageDecoder((img_size, img_size),scale=scale),transforms=image_pipeline),             
    } 
    pipelines['label'] = label_pipeline
    return pipelines

def build_ffcv_loader(args):
    train_transform = ThreeAugmentPipeline(img_size=args.input_size,device=args.device)
    val_transform = ValPipeline(img_size=args.input_size,device=args.device)
    order = OrderOption.RANDOM if args.distributed else OrderOption.QUASI_RANDOM
    if args.ra>1:
        repeat = args.ra
        class RepeatedRandom(Random):
            def __init__(self, loader:'Loader'):
                super().__init__(loader)

            def sample_order(self, epoch: int):
                order = super().sample_order(epoch)
                new_order = []
                for idx in order:
                    new_order.extend([idx] * repeat)
                return new_order
                        
        order = RepeatedRandom
        
    data_loader_train =  Loader(args.train_path, pipelines=train_transform,batches_ahead=10,
                        batch_size=args.batch_size, num_workers=args.num_workers, 
                        order=order, distributed=args.distributed,seed=args.seed)
    

    data_loader_val =  Loader(args.val_path, pipelines=val_transform,
                        batch_size=args.batch_size, num_workers=args.num_workers, batches_ahead=10,
                        distributed=args.distributed,seed=args.seed)
    print("Load dataset:", data_loader_train)
    return data_loader_train, data_loader_val
