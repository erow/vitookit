"""
Channel-wise Vision Transformer (CvT) models.
References:
- CvT: Introducing Convolutions to Vision Transformers
"""

from vitookit.utils.attention import SpatioTemporalBlock
from timm.models.vision_transformer import PatchEmbed, VisionTransformer
import torch
import torch.nn as nn


class CViT(nn.Module):
    """Channel-wise Vision Transformer (CViT) model."""

    def __init__(
        self,
        channels=4,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=0.0,
        sep_channel_attn=False,
    ):
        super(CViT, self).__init__()
        self.embed_dim = embed_dim
        self.channels = channels
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim*channels,
            strict_img_size=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, channels, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 1 + self.patch_embed.num_patches, embed_dim))
        self.channel_embed = nn.Parameter(torch.zeros(1, channels, 1 , embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(embed_dim, num_heads, channels if sep_channel_attn else -1, 
                                    dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(channels*embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        

    def patchify(self, x):
        
        x = self.patch_embed(x) # (B, num_patches, channels * embed_dim)
        B,N,_ = x.shape
        x = x.view(B, N, self.channels, self.embed_dim) # (B, num_patches, channels, embed_dim)
        x = x.permute(0,2,1,3) # (B, channels, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1, -1)  # (B, channels, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=2)
        x = x + self.pos_embed
        x = x + self.channel_embed
        x = self.pos_drop(x)
        return x
        
        
    def forward_features(self, x):
        x = self.patchify(x)  # (B, channels, num_patches + 1, embed_dim)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x
        
    def forward_head(self, x):
        B, C, N, D = x.shape
        cls_output = x[:, :, 0].flatten(1)  # (B, channels * embed_dim)
        logits = self.head(cls_output)
        return logits
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
from timm.models import register_model, build_model_with_cfg

def _create_model(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model_cls = CViT
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        # pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        # feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model

@register_model
def cvit_tiny_4ch(**kwargs):
    """CViT-Tiny model with 4 channels."""
    default_cfg = dict(
        channels=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        patch_size=4,
        img_size=32,
    )
    default_cfg.update(kwargs)
    model = _create_model('cvit_tiny_4ch', **default_cfg)
    return model

@register_model
def cvit_nano_4ch(**kwargs):
    """CViT-Nano model with 4 channels."""
    default_cfg = dict(
        channels=4,
        embed_dim=48,
        depth=12,
        num_heads=3,
        patch_size=4,
        img_size=32,
    )
    default_cfg.update(kwargs)
    model = _create_model('cvit_nano_4ch', **default_cfg)
    return model


if __name__ == "__main__":
    model = cvit_tiny_4ch().cuda()
    x = torch.randn(1, 3, 32, 32).cuda()
    logits = model(x)
    print(logits.shape)  # Expected output: torch.Size([1, 1000])