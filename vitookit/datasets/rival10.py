"""RIVAL10 dataset.
summary: 26k high resolution images organized into 10 classes with 18 informative visual attributes, as well as segmentation masks for each attribute.
paper: https://arxiv.org/pdf/2201.10766
code: https://github.com/mmoayeri/RIVAL10/tree/gh-pages
"""

from io import BytesIO
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import os
import glob
import json
import random
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
import pickle
from binascii import a2b_base64
from tqdm import tqdm

# REDACTED FOR ANONONYMITY


class RIVAL10(Dataset):
    def __init__(self, root, IN1K_root, train=True, return_masks=False):
        '''
        When return_masks is True, the object mask is returned, as well as the image and label.

        Use LocalRIVAL10 with masks_dict=True to obtain attribute masks. See local_rival10.py.
        '''
        self._PARTITIONED_URLS_DICT_PATH = f'{root}/train_test_split_by_url.json'
        self._LABEL_MAPPINGS = f'{root}/label_mappings.json'
        self._WNID_TO_CLASS = f'{root}/wnid_to_class.json'
        self._MASKS_TEMPLATE_ROOT = os.path.join(root, '{}/entire_object_masks/')
        self._NO_EO_MASKS_PATH = None
        self._IMAGENET_ROOT = IN1K_root

        self.train = train
        self.return_masks = return_masks
        self.mask_root = self._MASKS_TEMPLATE_ROOT.format('train' if self.train else 'test')
        self.instances = self.collect_instances()
        self.resize = transforms.Resize((224,224))

    def collect_instances(self):
        with open(self._PARTITIONED_URLS_DICT_PATH, 'r') as f:
            train_test_split = json.load(f)

        if self.train:
            urls = train_test_split['train']
        else:
            urls = train_test_split['test']
        
        with open(self._LABEL_MAPPINGS, 'r') as f:
            label_mappings = json.load(f)
        with open(self._WNID_TO_CLASS, 'r') as f:
            wnid_to_class = json.load(f)

        wnids = [url.split('/')[-2] for url in urls]
        inet_class_names = [wnid_to_class[wnid] for wnid in wnids]
        dcr_idx = [label_mappings[class_name][1] for class_name in inet_class_names]

        # original urls correspond to S3 links. We want to access saved ImageNet copy in cml
        local_urls = [self._IMAGENET_ROOT + url.replace('https://feizi-lab-datasets.s3.us-east-2.amazonaws.com/imagenet/','') for url in urls]
        
        instances = dict({i:(url, dcr_ind) for i,(url, dcr_ind) in enumerate(zip(local_urls, dcr_idx))})
        return instances
    
    def __len__(self):
        return len(self.instances)

    def transform(self, imgs):
        transformed_imgs = []
        i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.8,1.0),ratio=(0.75,1.25))
        coin_flip = (random.random() < 0.5)
        for ind, img in enumerate(imgs):
            if self.train:
                img = TF.crop(img, i, j, h, w)

                if coin_flip:
                    img = TF.hflip(img)

            img = TF.to_tensor(self.resize(img))
            
            if img.shape[0] == 1:
                img = torch.cat([img, img, img], axis=0)
            
            transformed_imgs.append(img)

        return transformed_imgs

    def __getitem__(self, i):
        url, label = self.instances[i]
        img = Image.open(url)
        if img.mode == 'L':
            img = np.array(img)
            img = np.stack([img, img, img], axis=-1) 
            img = Image.fromarray(img)
        imgs = [img]
        if self.return_masks:
            wnid, fname = url.split('/')[-2:]
            mask_path = self.mask_root + wnid + '_' + fname
            mask = Image.open(mask_path)
            imgs.append(mask)

        imgs = self.transform(imgs)
        if self.return_masks:
            return imgs[0], imgs[1], label
        else:
            return imgs[0], label

_ALL_ATTRS = [
    'long-snout', 'wings', 'wheels', 'text', 'horns', 'floppy-ears', 'ears', 'colored-eyes', 'tail', 'mane',
    'beak', 'hairy',  'metallic', 'rectangular', 'wet', 
    'long', 'tall', 'patterned']

def attr_to_idx(attr):
    return _ALL_ATTRS.index(attr)

def idx_to_attr(idx):
    return _ALL_ATTRS[idx]

def resize(img): 
    return np.array(Image.fromarray(np.uint8(img)).resize((224,224))) / 255

def to_3d(img):
    return np.stack([img, img, img], axis=-1)

def save_uri_as_img(uri, ):
    ''' saves raw mask and returns it as an image'''
    binary_data = a2b_base64(uri)
    img_data = BytesIO(binary_data)
    img = mpimg.imread(img_data)
    img = resize(img)
    # binarize mask
    img = np.sum(img, axis=-1)
    img[img != 0] = 1
    img = to_3d(img)
    return img

class LocalRIVAL10(Dataset):
    def __init__(self, root, train=True, masks_dict=True,img_size=224):
        '''
        Set masks_dict to be true to include tensor of attribute segmentations when retrieving items.

        See __getitem__ for more documentation. 
        '''
        _DATA_ROOT = os.path.join(root, '{}')
        self._LABEL_MAPPINGS = f'{root}/label_mappings.json'
        self._WNID_TO_CLASS = f'{root}/wnid_to_class.json'
        self.train = train
        self.data_root = _DATA_ROOT.format('train' if self.train else 'test')
        self.masks_dict = masks_dict

        self.instance_types = ['ordinary']        
        
        self.instances = self.collect_instances()
        self.resize = transforms.Resize((img_size,img_size))

        with open(self._LABEL_MAPPINGS, 'r') as f:
            self.label_mappings = json.load(f)
        with open(self._WNID_TO_CLASS, 'r') as f:
            self.wnid_to_class = json.load(f)

    def get_rival10_og_class(self, img_url):
        wnid = img_url.split('/')[-1].split('_')[0]
        inet_class_name = self.wnid_to_class[wnid]
        classname, class_label = self.label_mappings[inet_class_name]
        return classname, class_label

    def collect_instances(self):
        self.instances_by_type = dict()
        self.all_instances = []
        for subdir in self.instance_types:
            instances = []
            dir_path = os.path.join(self.data_root, subdir)
            for f in tqdm(glob.glob(dir_path+'/*')):
                if '.JPEG' in f and 'merged_mask' not in f:
                    img_url = f
                    label_path = f[:-5] + '_attr_labels.npy'
                    merged_mask_path = f[:-5] + '_merged_mask.JPEG'
                    mask_dict_path = f[:-5] + '_attr_dict.pkl'
                    instances.append((img_url, label_path, merged_mask_path, mask_dict_path))
            self.instances_by_type[subdir] = instances.copy()
            self.all_instances.extend(self.instances_by_type[subdir])

    def __len__(self):
        return len(self.all_instances)

    def transform(self, imgs):
        transformed_imgs = []
        i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.8,1.0),ratio=(0.75,1.25))
        coin_flip = (random.random() < 0.5)
        for ind, img in enumerate(imgs):
            if self.train:
                img = TF.crop(img, i, j, h, w)

                if coin_flip:
                    img = TF.hflip(img)

            img = TF.to_tensor(self.resize(img))
            
            if img.shape[0] == 1:
                img = torch.cat([img, img, img], axis=0)
            
            transformed_imgs.append(img)

        return transformed_imgs

    def merge_all_masks(self, mask_dict):
        merged_mask = np.zeros((224,224,3))
        for attr in mask_dict:
            if attr == 'entire-object':
                continue
            mask_uri = mask_dict[attr]
            mask = save_uri_as_img(mask_uri)
            merged_mask = mask if merged_mask is None else mask + merged_mask
        merged_mask[merged_mask > 0] = 1
        return merged_mask

    def __getitem__(self, i):
        '''
        Returns dict with following keys:
            img
            attr_labels: binary vec with 1 for present attrs
            changed_attr_labels: binary vec with 1 for attrs that were removed or pasted (not natural)
            merged_mask: binary mask with 1 for any attribute region
            attr_masks: tensor w/ mask per attribute. Masks are empty for non present attrs
        '''
        img_url, label_path,  merged_mask_path, mask_dict_path = self.all_instances[i]

        # get rival10 info for original image (label may not hold for attr-augmented images)
        class_name, class_label = self.get_rival10_og_class(img_url)

        # load img
        img = Image.open(img_url)
        if img.mode == 'L':
            img = img.convert("RGB")

        # load labels
        labels = np.load(label_path)
        attr_labels = torch.Tensor(labels[0]).long()
        changed_attrs = torch.Tensor(labels[1]).long() # attrs that were added or removed

        merged_mask_img = Image.open(merged_mask_path)
        imgs = [img, merged_mask_img]
        if self.masks_dict:
            try:
                with open(mask_dict_path, 'rb') as fp:
                    mask_dict = pickle.load(fp)
            except:
                mask_dict = dict()
            for attr in mask_dict:
                mask_uri = mask_dict[attr]
                mask = save_uri_as_img(mask_uri)
                imgs.append(Image.fromarray(np.uint8(255*mask)))
        
        transformed_imgs = self.transform(imgs)
        img = transformed_imgs.pop(0)
        merged_mask = transformed_imgs.pop(0)
        out = dict({'img':img, 
                    'attr_labels': attr_labels, 
                    'changed_attrs': changed_attrs,
                    'merged_mask' :merged_mask,
                    'og_class_name': class_name,
                    'og_class_label': class_label})
        if self.masks_dict:
            attr_masks = [torch.zeros(img.shape) for i in range(len(_ALL_ATTRS)+1)]
            for i, attr in enumerate(mask_dict):
                # if attr == 'entire-object':
                ind = -1 if attr == 'entire-object' else attr_to_idx(attr)
                attr_masks[ind] = transformed_imgs[i]
            out['attr_masks'] = torch.stack(attr_masks)
        
        return out