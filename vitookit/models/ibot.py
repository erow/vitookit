from timm.models.vision_transformer import _cfg, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, generate_default_cfgs
from timm.models.vision_transformer import VisionTransformer, build_model_with_cfg, checkpoint_filter_fn
from timm.models._pretrained import PretrainedCfg
from timm.models._registry import _model_pretrained_cfgs, _model_has_pretrained, _model_with_tags, register_model
from dataclasses import replace
from typing import Optional

from functools import partial

default_cfgs={
    'vit_base_patch16_224.ibot': _cfg(
        url='https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth',
        custom_load=True, 
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_small_patch16_224.ibot': _cfg(
        url='https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    # 'ibot_small_patch16_224.in1k': _cfg(
    #     url='https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth',
    #     mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    # 'ibot_base_patch16_224.in1k': _cfg(
    #     url='https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth',
    #     mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    # 'ibot_large_patch16_224.in1k': _cfg(
    #     url='https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/checkpoint_teacher.pth',
    #     mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    # 'ibot_base_patch16_224.in1krand': _cfg(
    #     url='https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/checkpoint_teacher.pth',
    #     mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    # 'ibot_large_patch16_224.in1krand': _cfg(
    #     url='https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_rand_mask/checkpoint_teacher.pth',
    #     mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
}

def _create_ibot_transformer(
        variant: str,
        pretrained: bool = False,
        **kwargs,
) -> VisionTransformer:
    # Check if we should use NaFlexVit instead
    
    out_indices = kwargs.pop('out_indices', 3)
    _filter_fn = checkpoint_filter_fn


    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=False,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    
def add_extra_vit_configs(default_cfgs):
    """Register extra ViT model configurations into timm's default_cfgs."""
    for model_name, default_cfg in default_cfgs.items():
        for tag_idx, tag in enumerate(default_cfg.tags):
            is_default = tag_idx == 0
        pretrained_cfg = default_cfg.cfgs[tag]
        model_name_tag = '.'.join([model_name, tag]) if tag else model_name
        replace_items = dict(architecture=model_name, tag=tag if tag else None)
        if pretrained_cfg.hf_hub_id and pretrained_cfg.hf_hub_id == 'timm/':
            # auto-complete hub name w/ architecture.tag
            replace_items['hf_hub_id'] = pretrained_cfg.hf_hub_id + model_name_tag
        pretrained_cfg = replace(pretrained_cfg, **replace_items)

        if is_default:
            _model_pretrained_cfgs[model_name] = pretrained_cfg
            if pretrained_cfg.has_weights:
                # add tagless entry if it's default and has weights
                _model_has_pretrained.add(model_name)

        if tag:
            _model_pretrained_cfgs[model_name_tag] = pretrained_cfg
            if pretrained_cfg.has_weights:
                # add model w/ tag if tag is valid
                _model_has_pretrained.add(model_name_tag)
            _model_with_tags[model_name].append(model_name_tag)
        else:
            _model_with_tags[model_name].append(model_name)  # has empty tag (to slowly remove these instances)

default_cfgs = generate_default_cfgs(default_cfgs)
add_extra_vit_configs(default_cfgs)

# @register_model
# def ibot_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
#     """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
#     model = _create_ibot_transformer('ibot_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model

# import timm
# timm.create_model('vit_small_patch16_224.ibot', pretrained=True)  # test loading