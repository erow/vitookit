from functools import partial
from typing import Dict, List, Optional, Set
import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import checkpoint_filter_fn, Final, use_fused_attn, PatchEmbed,Mlp,_cfg
from timm.layers import AttentionPoolLatent, resample_abs_pos_embed,trunc_normal_
from timm.models.layers import DropPath, to_2tuple
from types import MethodType
import numpy as np


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]    
        
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
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



class AttentionPoolLatent(nn.Module):
    """ Attention pooling w/ latent query
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 8,
            feat_size: Optional[int] = None,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            latent_len: int = 1,
            latent_dim: int = None,
            pos_embed: str = '',
            pool_type: str = 'token',
            norm_layer: Optional[nn.Module] = None,
            act_layer: Optional[nn.Module] = nn.GELU,
            drop: float = 0.0,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.fused_attn = use_fused_attn()

        if pos_embed == 'abs':
            assert feat_size is not None
            self.pos_embed = nn.Parameter(torch.zeros(feat_size, in_features, requires_grad=True))
        else:
            self.pos_embed = None

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim, requires_grad=True))

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        if qk_norm:
            qk_norm_layer = norm_layer or nn.LayerNorm
            self.q_norm = qk_norm_layer(self.head_dim)
            self.k_norm = qk_norm_layer(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), act_layer=act_layer)

        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        trunc_normal_(self.latent, std=self.latent_dim ** -0.5)
        
    def forward(self, x, idx):
        B, N, C = x.shape

        q_latent = self.latent[:,idx:idx+1].expand(B, -1, -1)
        q = self.q(q_latent).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        return x



def build_sincos1d_pos_embed(
        L: int,
        dim: int = 64,
        temperature: float = 10000.,
        reverse_coord: bool = False,
        interleave_sin_cos: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate 1D sine-cosine positional embeddings.

    Args:
        feat_shape (List[int]): Shape of the feature map (e.g., [L]).
        dim (int): Dimension of the positional embeddings. Must be even.
        temperature (float): Temperature factor in the positional encoding formula.
        reverse_coord (bool): Whether to reverse the coordinate direction.
        interleave_sin_cos (bool): Whether to interleave sine and cosine components.
        dtype (torch.dtype): Data type of the output tensor.
        device (Optional[torch.device]): Device on which to place the tensor.

    Returns:
        torch.Tensor: Positional embeddings of shape (L, dim).
    """
    if dim % 2 != 0:
        raise ValueError("dim must be an even number")

    half_dim = dim // 2
    embeddings = torch.zeros((L, dim), dtype=dtype, device=device)

    # Position along the sequence
    position = torch.arange(L, dtype=dtype, device=device).unsqueeze(1)

    # Compute the sine and cosine components
    # exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    exp = torch.exp(torch.arange(half_dim, dtype=dtype, device=device)  / half_dim)
    div_term = 1. / (temperature ** exp)
    if reverse_coord:
        div_term = div_term.flip(0)

    if interleave_sin_cos:
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
    else:
        embeddings[:, :half_dim] = torch.sin(position * div_term)
        embeddings[:, half_dim:] = torch.cos(position * div_term)

    return embeddings


@gin.configurable    
class MultiScaleVisionTransformer(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone    
    """
    def __init__(self, shared_embed=True, causal=True, shared_head=True,
                 img_sizes=[112,224], patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 block_layers = Block,
                 act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                 init_scale=1e-4,
                 **kwargs):
        print(" Warn: unknow args", kwargs)
        super().__init__()
        self.embed_dim = embed_dim
        img_size=img_sizes[-1]
        self.img_size=img_size
        num_frames = len(img_sizes)
        self.img_sizes = img_sizes
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.shared_embed = shared_embed
        self.shared_head = shared_head
            
        num_patches = (img_size // patch_size) * (img_size // patch_size) 
        # shared patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim,  output_fmt='NHWC',strict_img_size=False)
        if not shared_embed:                
            self.patch_embeds = nn.ModuleList([
                PatchEmbed(s, patch_size, in_chans, embed_dim,  output_fmt='NHWC', strict_img_size=False) 
                for s in img_sizes[:-1]])
        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=True)         
        torch.nn.init.normal_(self.pos_embed, std=.02)
        # add temporal information
        self.temporal_pos_embed = nn.Parameter(torch.rand(1, num_frames,1, embed_dim), requires_grad=True)
        self.temporal_pos_embed.data[:] = build_sincos1d_pos_embed(num_frames,embed_dim).reshape_as(self.temporal_pos_embed)
        # torch.nn.init.normal_(self.temporal_pos_embed, std=.02)
        
        dpr = [drop_path_rate for i in range(depth)]
        
        self.blocks = nn.ModuleList([
            block_layers(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        
        # set up attention mask
        token_list = [1 + int(s//patch_size )**2 for s in img_sizes]
        token_accum = [0] + [np.sum(token_list[:i+1]) for i in range(len(token_list))]
        self.token_accum = token_accum
        
        if causal:
            attn_mask = torch.zeros((token_accum[-1], token_accum[-1]), dtype=torch.bool)
            
            # set visible for current frame and cls token
            for i in range(num_frames):
                cur_token = token_accum[i+1]
                attn_mask[token_accum[i]:cur_token, :cur_token] = True 
                
            # hack for attention mask
            def attnmask_forward(self, x: torch.Tensor) -> torch.Tensor:
                B, N, C = x.shape
                # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)                
                q, k, v = qkv.unbind(0)
                # q, k = self.q_norm(q), self.k_norm(k)
                
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask = attn_mask.to(x.device),
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    scale=self.scale,
                )            

                x = x.transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x
            for name, module in self.blocks.named_modules():
                if isinstance(module, Attention):
                    module.forward = MethodType(attnmask_forward,module)                
        
        if not shared_head:
            self.heads= nn.ModuleList([nn.Linear(embed_dim,num_classes)
                            for i in range(num_frames)])
            self.attn_pool = None     
        else:
            self.head = nn.Linear(embed_dim,num_classes)
            
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,  latent_len=num_frames,
                num_heads=12,        
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
                

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'temporal_pos_embed','dist_token', 'attn_pool.latent'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
            head=r'^head|attn_pool|temporal_pos_embed',  # head and attention pooling
        )
    
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:        
        
        B, H, W, C = x.shape
        prev_grid_size = self.patch_embed.grid_size
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=(H, W),
            old_size=prev_grid_size,
            num_prefix_tokens=0,
        )
        x = x.view(B, -1, C)

        to_cat = [self.cls_token.expand(x.shape[0], -1, -1), 
                  x+pos_embed]
        x = torch.cat(to_cat, dim=1)
        return x
    
    def forward_features(self, imgs, pos_embed=None):
        B, C, H, W = imgs.shape
        
        tokens = []
        
        for i,s in enumerate(self.img_sizes):
            x = F.interpolate(imgs,size=s, mode='bilinear', align_corners=False)
            # TODO: not sharing patch    x = self.patch_embeds[i](x) 
            if self.shared_embed:
                x = self.patch_embed(x)
            else:
                if s == self.img_size:
                    x = self.patch_embed(x)
                else:
                    x = self.patch_embeds[i](x)
            x = self._pos_embed(x) # add cls_token and pos_embed
            x = x + self.temporal_pos_embed[:,i,:,:] 
            
            tokens+= [x]
        
        x = torch.cat(tokens, dim=1) # [B,L,D]
        # x = self.norm_pre(x)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    
    def pool(self, z, step):
        """ Pool the features of the input images.
        Args:
            z (torch.Tensor): features of the input images.
        Returns:
            torch.Tensor: pooled features.
        """
        # z = z[:,0]  # cls tolen
        if self.attn_pool is not None:
            z = self.attn_pool(z, step)  # attention pooling
        else:
            z = z[:,1:].mean()
        return z

    def forward(self, imgs,**kwargs):
        z = self.forward_features(imgs)
        preds = []
        distl = []
        pred = 0
        for i in range(self.num_frames):
            l = self.token_accum[i]
            r = self.token_accum[i+1]
            # if self.shared_head:
            #     head = self.head
            # else:
            #     head = self.heads[i]
            # pool 
            latent = self.pool(z[:,l:r], i)
            # distl.append(self.distl_head(z[:,l+1]))
            logit = self.head(latent) 
            # logit = head(z[:,l])
            preds.append(logit)
            pred += logit
        pred /= self.num_frames
        if self.training:        
            return preds[-1]
        else:
            return preds[-1]

from timm.models import register_model, build_model_with_cfg

def _create_mvit(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model_cls = MultiScaleVisionTransformer
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model

@register_model
def mvit_models(**kwargs):
    return MultiScaleVisionTransformer(**kwargs)

@register_model
def mvit2_tiny_patch2(**kwargs):
    model = _create_mvit(
        "mvit2_tiny_patch2",
        img_sizes = [16, 32], patch_size=2, embed_dim=192, depth=12, heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    
    return model
    
@register_model
def mvit2_tiny_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,   **kwargs):
    model = mvit_models(
         patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    
    return model
    
    
@register_model
def mvit2_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = mvit_models(
         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        print(model.load_state_dict(checkpoint["model"],False))

    return model

@register_model
def mvit2_medium_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False, **kwargs):
    model = mvit_models(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_medium_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        print(model.load_state_dict(checkpoint["model"],False))
    return model 

@register_model
def mvit2_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = mvit_models(
         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        print(model.load_state_dict(checkpoint["model"],False))
    return model

@register_model
def mvit3_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = mvit_models(
        img_sizes=[112, 192, 224], shared_embed=True, causal=False,
         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        print(model.load_state_dict(checkpoint["model"],False))
    return model

@register_model
def mvit2_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = mvit_models(
         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_large_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        print(model.load_state_dict(checkpoint["model"],False))
    return model
    
@register_model
def mvit2_huge_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = mvit_models(
         patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_huge_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k_v1.pth'
        else:
            name+='1k_v1.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        print(model.load_state_dict(checkpoint["model"],False))
    return model

if __name__ == "__main__":
    model = mvit_models(img_sizes=[16,32],patch_size=2,embed_dim=192)
    imgs = torch.randn(10,3,32,32)
    out = model(imgs)
    print(out.shape)