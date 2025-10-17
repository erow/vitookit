
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_

from timm.models.vision_transformer import Block, PatchEmbed, _cfg, vit_tiny_patch16_224, use_fused_attn
from timm.models.registry import register_model
from timm.models.cait import _create_cait, Cait, Mlp
from timm.models import build_model_with_cfg

import gin

class UpPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 type=0):
        super().__init__()
        self.embedding = PatchEmbed(img_size=img_size*2, patch_size=patch_size*2, in_chans=in_chans, embed_dim=embed_dim)
        self.type = type
        if type == 0:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        elif type == 1:
            self.up = nn.ConvTranspose2d(in_chans, in_chans, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif type == 2:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, x):
        x = self.up(x)
        x = self.embedding(x)
        return x
    
class SelfAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)


        self.proj_drop = nn.Dropout(proj_drop)
        self.fused_attn = use_fused_attn()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v,scale=self.scale)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)        

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FilteredSelfAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.filter = nn.Linear(dim, dim, bias=False)
        nn.utils.weight_norm(self.filter)
        c = dim//2
        self.filter.weight_v.data[:c,:c] = torch.eye(c)
        

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)


        self.proj_drop = nn.Dropout(proj_drop)
        self.fused_attn = use_fused_attn()

    def forward(self, x):
        x = self.filter(x) # apply filter to the input
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v,scale=self.scale)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)        

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StochasticallySkippedBlock(nn.Module):
    """
    A Transformer block that can be stochastically skipped during training.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, survival_prob=1.0):
        super().__init__()
        self.survival_prob = survival_prob

        # Standard Transformer Block components
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        # The core logic for your idea is here
        if not self.training or torch.rand(1).item() < self.survival_prob:
            # --- If we run the block ---
            # Either it's inference time (self.training is False)
            # Or our random number was less than the survival probability

            # 1. Self-Attention part
            attn_output, _ = self.self_attn(src, src, src)
            src = src + self.dropout1(attn_output)
            src = self.norm1(src)

            # 2. Feed-forward part
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(ff_output)
            src = self.norm2(src)
            
            # During inference, scale the output to account for the skipping
            if not self.training:
                src = src * self.survival_prob
        
        # If the block is skipped (in training), src is passed through unchanged
        return src
    
class ThinkVIT(Cait):        

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        if self.training:
            order = torch.randperm(len(self.blocks_token_only))[::2]
            for i, block_idx in enumerate(order):
                blk = self.blocks_token_only[block_idx]
                cls_tokens = blk(x, cls_tokens)
        else:
            for blk in self.blocks_token_only:
                cls_tokens = blk(x, cls_tokens)
                
        cls_tokens = self.head_drop(cls_tokens)
        cls_tokens = cls_tokens[:, 0]
        return self.head(cls_tokens)

def _create_thinkvit(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model = build_model_with_cfg(
        ThinkVIT,
        variant,
        pretrained,
        pretrained_filter_fn=None,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    
    return model

@register_model
def vit_tiny_up(pretrained=False, **kwargs):
    model = vit_tiny_patch16_224(pretrained=pretrained, **kwargs)
    patch_embed: PatchEmbed = model.patch_embed
    up_patch_embed = UpPatchEmbed(img_size=patch_embed.img_size[0], patch_size=patch_embed.patch_size[0],
                                 embed_dim=model.embed_dim)
    model.patch_embed = up_patch_embed
    return model

@register_model
def cait_xxt12_32(pretrained=False, **kwargs):
    model_args = dict(img_size=32,patch_size=2, embed_dim=192, depth=12, num_heads=4, init_values=1e-5,
                      attn_block=SelfAttn,depth_token_only=2,
                      )
    model = _create_cait('cait_xxt12_32', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def fit_s12c2_32(pretrained=False, **kwargs):
    model_args = dict(img_size=32,patch_size=2, embed_dim=192, depth=12, num_heads=4, init_values=1e-5,
                      attn_block=FilteredSelfAttn,depth_token_only=2,
                      )
    model = _create_cait('fit_s12c2_32', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def thinkvit_s12c4_32(pretrained=False, **kwargs):
    model_args = dict(img_size=32,patch_size=2, embed_dim=192, depth=12, num_heads=4, init_values=1e-5,
                      attn_block=SelfAttn, depth_token_only=4,
                      )
    model = _create_thinkvit('thinkvit_s12c4_32', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


class MoEBlock(nn.Module):
    def __init__(self, 
            in_features,
            num_experts=4,
            **kwargs):
        super().__init__()
        self.num_experts = num_experts
        # kwargs['hidden_features'] = kwargs.get('hidden_features', in_features*4)//num_experts
        self.experts = nn.ModuleList([Mlp(in_features, **kwargs) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.constant_(self.gate.bias, 0)
        self.register_buffer('gate_center', torch.zeros(1, 1, num_experts), persistent=True)
        self.momentum = 0.9

    def forward(self, x):
        B, N, C = x.shape
        gate_scores = self.gate(x)  
        
        p_experts = F.softmax(gate_scores-self.gate_center, dim=-1)  # Shape: (B, N, num_experts)
        ## Update gate center
        if self.training:
            self.gate_center.data[:] = self.momentum * self.gate_center + (1 - self.momentum) * gate_scores.detach().mean(dim=(0, 1), keepdim=True)

        experts = p_experts.argmax(dim=-1).flatten()  # Shape: (B N)
        x = x.reshape(B*N, C)
        dtype = x.dtype
        expert_outputs = torch.zeros_like(x,device=x.device).to(dtype)  # Shape: (B N, C)
        for i in range(self.num_experts):
            mask = (experts == i)
            if mask.sum() > 0:
                expert_outputs[mask] = self.experts[i](x[mask]).to(dtype)

        expert_outputs = expert_outputs.reshape(B, N, C)
        return expert_outputs
    
@register_model
def vmoe_xxt12_32(pretrained=False, num_experts=4, **kwargs):
    mlp_block = partial(MoEBlock, num_experts=num_experts)
    model_args = dict(img_size=32,patch_size=2, embed_dim=192, depth=12, num_heads=4, init_values=0,
                      attn_block=SelfAttn,depth_token_only=2,
                      mlp_block=mlp_block,
                      )
    model = _create_cait('vmoe_xxt12_32', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

if __name__ == "__main__":
    model =  vmoe_xxt12_32()
    x = torch.randn(10, 3, 32, 32)
    y = model(x)