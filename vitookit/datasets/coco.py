""" This file contains the code to load the COCO dataset. The data are described in https://cocodataset.org/#format-data. The labels are described in https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/. 
The return value of the __getitem__ function is a tuple of (image, mask). The image is a tensor of shape (3,480,480) and the mask is a tensor of shape (480,480). The mask is a class mask if mask_type is 'class' and an instance mask if mask_type is 'instance'. The class mask is a tensor of shape (480,480) where each pixel is the class id of the object that the pixel belongs to. The instance mask is a tensor of shape (480,480) where each pixel is the instance id of the object that the pixel belongs to. The instance id is the id field in the annotation. The class id is the category_id field in the annotation.
Usage: 
```
from vitookit.datasets.coco import COCO
data = COCO(image_root,anno_file,transform=tfms)
``` 
"""
import cv2
from typing import Any, Callable, Optional
from pycocotools import mask as maskUtils
from torchvision import datasets
import torch,math,os,torchvision,random, einops
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as tfms
import torchvision.transforms.functional as tfF

def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask( ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m

labels= 'person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush'.split(',')

class COCO(datasets.CocoDetection):
    def __init__(self, root: str, annFile: str, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, transforms: Callable[..., Any] | None = None, mask_type='class') -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        assert mask_type in ['class','instance']
        self.mask_type=mask_type
        
        self.classes = labels
        self.class_dict= {idx+1:label for idx,label in enumerate(self.classes)}
        
        
    def __getitem__(self, index: int):
        img,annotations= super().__getitem__(index)
        _, width,height = img.shape
        
        instance_masks = [np.zeros((width,height))]
        class_masks = [np.zeros((width,height))]
        ins_set = set()
        class_set = set()
        for annotation in annotations:
            m = annToMask(annotation, height,
                                width)
            m = cv2.resize(m,(width,height))
            
            # Some objects are so small that they're less than 3 patches
            # and end up rounded out. Skip those objects.
            
            ins_id = annotation['id']
            class_id = annotation['category_id']
            # The  id  field is a unique identifier assigned to each individual object in the dataset. 
            # On the other hand,  category_id  is the identifier for the category or class that the object belongs to.
            
            instance_masks.append(m*ins_id)
            class_masks.append(m*class_id)
            ins_set.add(ins_id)
            class_set.add(class_id)
        
        # merge the masks
        instance_mask = np.stack(instance_masks).max(0)
        class_mask = np.stack(class_masks).max(0)
        mask = class_mask if self.mask_type=='class' else instance_mask
        
        return img, mask

if __name__ == '__main__':


    from torchvision import datasets, transforms
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    
    common_tf = transforms.Compose([
        # transforms.CenterCrop(args.image_size),
        transforms.Resize((480,480)),
        ])
    
    tfms=transforms.Compose([
        common_tf,
        transforms.ToTensor(),
        # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    def denormalize(imagesToPrint):
        imagesToPrint = imagesToPrint.clone()
        imagesToPrint *= torch.Tensor(IMAGENET_DEFAULT_STD,device=imagesToPrint.device).reshape(3,1,1)
        imagesToPrint += torch.Tensor(IMAGENET_DEFAULT_MEAN,device=imagesToPrint.device).reshape(3,1,1)
        return imagesToPrint.clamp(0,1)
    
    #%% ## get representations
    nh,hw = 30,30
    data = COCO(         '/vol/research/datasets/still/MSCOCO/images/train2017',
                '/vol/research/datasets/still/MSCOCO/annotations/instances_train2017.json',transform=tfms)
    
    for idx in [29,19,3]:
        img,_ = data[idx]
        cv2.imwrite(f'outputs/debug/coco-{idx}.png',img.permute(1,2,0).numpy()*255)
# %%
