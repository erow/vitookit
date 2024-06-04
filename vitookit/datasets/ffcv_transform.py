"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
Copy from https://github.com/facebookresearch/FFCV-SSL/blob/6458e33f0753e7a35bc639517a763350a0fc2177/ffcv/transforms/colorjitter.py
"""


import numpy as np
from collections.abc import Sequence
from typing import Callable, Optional, Tuple
from dataclasses import replace
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
import numba as nb
import numbers
import math
import random
from numba import njit, jit
import gin
from cv2 import GaussianBlur
from scipy.ndimage import gaussian_filter

from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage,RandomHorizontalFlip, View, Convert
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder

import torch
import torchvision.transforms.v2 as tfms
from torchvision.transforms import functional as F
from torch import nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


@gin.configurable
def SimplePipeline(img_size=224,scale=(0.2,1), ratio=(3.0/4.0, 4.0/3.0)):
    image_pipeline = [
            RandomResizedCropRGBImageDecoder((img_size, img_size), scale=scale,ratio=ratio,),
            RandomHorizontalFlip(),            
            ToTensor(), 
            ToDevice(torch.device('cuda')),
            ToTorchImage(),
            Convert(torch.float16),
            tfms.Normalize(mean=[0.485*255, 0.456*255, 0.406*255], std=[0.229*255, 0.224*255, 0.225*255], inplace=True),
            ]
    label_pipeline = [IntDecoder(), ToTensor(),ToDevice(torch.device('cuda'))]
    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    } 
    return pipelines    

@gin.configurable
def ValPipeline(img_size=224,ratio= 224/256):
    image_pipeline = [
            CenterCropRGBImageDecoder((img_size, img_size), ratio),
            ToTensor(), 
            # ToDevice(torch.device('cuda')),        
            ToTorchImage(),
            Convert(torch.float32),
            tfms.Normalize(mean=[0.485*255, 0.456*255, 0.406*255], std=[0.229*255, 0.224*255, 0.225*255], inplace=True),
            ]
    label_pipeline = [IntDecoder(), 
                      ToTensor(),
                    #   ToDevice(torch.device('cuda')),
                      View(-1)]
    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    } 
    return pipelines  


@njit(parallel=False, fastmath=True, inline="always")
def apply_cj(
    im,
    apply_bri,
    bri_ratio,
    apply_cont,
    cont_ratio,
    apply_sat,
    sat_ratio,
    apply_hue,
    hue_factor,
):

    gray = (
        np.float32(0.2989) * im[..., 0]
        + np.float32(0.5870) * im[..., 1]
        + np.float32(0.1140) * im[..., 2]
    )
    one = np.float32(1)
    # Brightness
    if apply_bri:
        im = im * bri_ratio

    # Contrast
    if apply_cont:
        im = cont_ratio * im + (one - cont_ratio) * np.float32(gray.mean())

    # Saturation
    if apply_sat:
        im[..., 0] = sat_ratio * im[..., 0] + (one - sat_ratio) * gray
        im[..., 1] = sat_ratio * im[..., 1] + (one - sat_ratio) * gray
        im[..., 2] = sat_ratio * im[..., 2] + (one - sat_ratio) * gray

    # Hue
    if apply_hue:
        hue_factor_radians = hue_factor * 2.0 * np.pi
        cosA = np.cos(hue_factor_radians)
        sinA = np.sin(hue_factor_radians)
        v1, v2, v3 = 1.0 / 3.0, np.sqrt(1.0 / 3.0), (1.0 - cosA)
        hue_matrix = [
            [
                cosA + v3 / 3.0,
                v1 * v3 - v2 * sinA,
                v1 * v3 + v2 * sinA,
            ],
            [
                v1 * v3 + v2 * sinA,
                cosA + v1 * v3,
                v1 * v3 - v2 * sinA,
            ],
            [
                v1 * v3 - v2 * sinA,
                v1 * v3 + v2 * sinA,
                cosA + v1 * v3,
            ],
        ]
        hue_matrix = np.array(hue_matrix, dtype=np.float64).T
        for row in nb.prange(im.shape[0]):
            im[row] = im[row] @ hue_matrix
    return np.clip(im, 0, 255).astype(np.uint8)


class RandomColorJitter(Operation):
    """Add ColorJitter with probability jitter_prob.
    Operates on raw arrays (not tensors), ranging from 0 to 255.

    see https://github.com/pytorch/vision/blob/28557e0cfe9113a5285330542264f03e4ba74535/torchvision/transforms/functional_tensor.py#L165
     and https://sanje2v.wordpress.com/2021/01/11/accelerating-data-transforms/
    Parameters
    ----------
    jitter_prob : float, The probability with which to apply ColorJitter.
    brightness (float or tuple of float (min, max)): How much to jitter brightness.
        brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
        or the given [min, max]. Should be non negative numbers.
    contrast (float or tuple of float (min, max)): How much to jitter contrast.
        contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
        or the given [min, max]. Should be non negative numbers.
    saturation (float or tuple of float (min, max)): How much to jitter saturation.
        saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
        or the given [min, max]. Should be non negative numbers.
    hue (float or tuple of float (min, max)): How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(
        self,
        jitter_prob=0.5,
        brightness=0.8,
        contrast=0.4,
        saturation=0.4,
        hue=0.2,
        seed=None,
    ):
        super().__init__()
        self.jitter_prob = jitter_prob

        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5))
        self.seed = seed
        assert self.jitter_prob >= 0 and self.jitter_prob <= 1

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f"If {name} is a single number, it must be non negative."
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with length 2."
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            setattr(self, f"apply_{name}", False)
        else:
            setattr(self, f"apply_{name}", True)
        return tuple(value)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()

        jitter_prob = self.jitter_prob

        apply_bri = self.apply_brightness
        bri = self.brightness
        

        apply_cont = self.apply_contrast
        cont = self.contrast

        apply_sat = self.apply_saturation
        sat = self.saturation

        apply_hue = self.apply_hue
        hue = self.hue

        seed = self.seed
        if seed is None:

            def color_jitter(images, _):
                for i in my_range(images.shape[0]):
                    if np.random.rand() > jitter_prob:
                        continue

                    images[i] = apply_cj(
                        images[i].astype("float64"),
                        apply_bri,
                        np.random.uniform(bri[0], bri[1]),
                        apply_cont,
                        np.random.uniform(cont[0], cont[1]),
                        apply_sat,
                        np.random.uniform(sat[0], sat[1]),
                        apply_hue,
                        np.random.uniform(hue[0], hue[1]),
                    )
                return images

            color_jitter.is_parallel = True
            return color_jitter

        def color_jitter(images, _, counter):

            random.seed(seed + counter)
            N = images.shape[0]
            values = np.zeros(N)
            bris = np.zeros(N)
            conts = np.zeros(N)
            sats = np.zeros(N)
            hues = np.zeros(N)
            for i in range(N):
                values[i] = np.float32(random.uniform(0, 1))
                bris[i] = np.float32(random.uniform(bri[0], bri[1]))
                conts[i] = np.float32(random.uniform(cont[0], cont[1]))
                sats[i] = np.float32(random.uniform(sat[0], sat[1]))
                hues[i] = np.float32(random.uniform(hue[0], hue[1]))
            for i in my_range(N):
                if values[i] > jitter_prob:
                    continue
                images[i] = apply_cj(
                    images[i].astype("float64"),
                    apply_bri,
                    bris[i],
                    apply_cont,
                    conts[i],
                    apply_sat,
                    sats[i],
                    apply_hue,
                    hues[i],
                )
            return images

        color_jitter.is_parallel = True
        color_jitter.with_counter = True
        return color_jitter

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), None)

class Grayscale(Operation):
    """Add Gaussian Blur with probability blur_prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    blur_prob : float
        The probability with which to flip each image in the batch
        horizontally.
    """

    def __init__(self):
        super().__init__()

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        
        def grayscale(images, _):
            for i in my_range(images.shape[0]):
                images[i] = (
                    0.2989 * images[i, ..., 0:1]
                    + 0.5870 * images[i, ..., 1:2]
                    + 0.1140 * images[i, ..., 2:3]
                )
            return images
        grayscale.is_parallel = True
        return grayscale

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (previous_state, None)



class Solarization(Operation):
    """Solarize the image randomly with a given probability by inverting all pixel
    values above a threshold. If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Parameters
    ----------
        solarization_prob (float): probability of the image being solarized. Default value is 0.5
        threshold (float): all pixels equal or above this value are inverted.
    """

    def __init__(
        self, threshold: float = 128,
    ):
        super().__init__()
        self.threshold = threshold

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        threshold = self.threshold
        def solarize(images, _):
            for i in my_range(images.shape[0]):
                mask = images[i] >= threshold
                images[i] = np.where(mask, 255 - images[i], images[i])
            return images

        solarize.is_parallel = True
        return solarize

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # No updates to state or extra memory necessary!
        return previous_state, None
  

@njit
def generate_gaussian_filter(sigma: int | float,filter_shape: list | tuple = (3, 3)):
    # 'sigma' is the standard deviation of the gaussian distribution

    m, n = filter_shape
    m_half = m // 2
    n_half = n // 2

    # initializing the filter
    gaussian_filter = np.zeros((m, n), np.float32)
    k = 1 / (2.0 * sigma**2.0)
    # generating the filter
    for y in range(-m_half, m_half+1):
        for x in range(-n_half, n_half+1):
            exp_term = np.exp(-(x**2 + y**2) *k)
            gaussian_filter[y+m_half, x+n_half] = exp_term
    kernel = gaussian_filter/gaussian_filter.sum()
    return kernel

@njit
def convolution(image: np.ndarray, kernel: list | tuple, output: np.ndarray) -> np.ndarray:
    '''
    It is a "valid" Convolution algorithm implementaion.
    ### Example
    >>> import numpy as np
    >>> from PIL import Image
    >>>
    >>> kernel = np.array(
    >>>     [[-1, 0, 1],
    >>>     [-2, 0, 2],
    >>>     [-1, 0, 1]], np.float32
    >>> )
    >>> img = np.array(Image.open('./lenna.png'))
    >>> res = convolution(img, Kx)
    '''
    if len(image.shape) == 3:
        m_i, n_i, c_i = image.shape
    else:
        raise Exception('Shape of image not supported')

    m_k, n_k = kernel.shape

    y_strides = m_i - m_k + 1  # possible number of strides in y direction
    x_strides = n_i - n_k + 1  # possible number of strides in x direction

    pad_y = (m_k - 1) // 2
    pad_x = (n_k - 1) // 2
    output.fill(0)
    for i in range(y_strides):
        for j in range(x_strides):
            for c in range(c_i): # looping over the all channels
                for k_i in range(m_k):
                    for k_j in range(n_k):
                        output[i+pad_y,j+pad_x,c] += image[i+k_i,j+k_j,c] * kernel[k_i,k_j]
              
    return output


from torchvision.transforms import InterpolationMode

@gin.configurable
class RandDownSampling(nn.Module):
    def __init__(self, r=(0.25,0.75)) -> None:
        super().__init__()
        self.r = r
    def forward(self,x):
        h, w = x.shape[-2:]
        r = random.uniform(*self.r)
        if r >= 0.99:
            return x
        nh,hw = int(h*r),int(w*r)
        down = F.resize(x,(nh,hw),interpolation=InterpolationMode.BICUBIC)
        up = F.resize(down,(h,w),interpolation=InterpolationMode.BICUBIC)
        return up

class ThreeAugmentation(nn.Module):
    """Apply single transformation randomly picked from a list. This transform does not support torchscript."""

    def __init__(self, ):
        super().__init__()
        self.guassian_blur = tfms.GaussianBlur(5,sigma=(0.1,2))
        

    def __call__(self, x):
        op_index = torch.randint(0,3,(len(x),))
        
        for i,op in enumerate([self.guassian_blur,
                               lambda x: F.solarize(x,0),
                               F.rgb_to_grayscale]):
            tf_mask = op_index == i
            x[tf_mask] = op(x[tf_mask])
        return x

    def __repr__(self) -> str:
        return f"{super().__repr__()}(p={self.p})"
        
@gin.configurable
def ThreeAugmentPipeline(img_size=224,scale=(0.08,1), color_jitter=None):
    """
    ThreeAugmentPipeline
    """
    image_pipeline = (
        # first_tfl 
        [   RandomResizedCropRGBImageDecoder((img_size, img_size), scale=scale,),
            RandomHorizontalFlip(),]+
        # second_tfl
        (   [RandomColorJitter(jitter_prob=0.5, brightness=color_jitter, contrast=color_jitter, saturation=color_jitter,)] if color_jitter else []) + 
        # final_tfl
        [
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
            ToTensor(), 
            # ToDevice(torch.device('cuda')),        
            ToTorchImage(),
            ThreeAugmentation(),
        ]) 
        
    label_pipeline = [IntDecoder(), ToTensor(),View(-1)]
    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    } 
    return pipelines   