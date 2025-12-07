from __future__ import annotations

from math import pi
from typing import Literal, Union

import torch
from einops import rearrange, repeat
from torch import nn, einsum, broadcast_tensors, Tensor
from torch.amp import autocast
from torch.nn import Module
import torch.nn.functional as F
if False:
# if hasattr(F,"scaled_dot_product_attention"):
    scaled_dot_product_attention = F.scaled_dot_product_attention
else:
    def scaled_dot_product_attention(
            query: Tensor,
            key: Tensor,
            value: Tensor,
            is_causal: bool = False,
            scale = None
    ) -> Tensor:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query)
        if is_causal:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(attn_bias)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if scale is None:
            scale = query.shape[-1]**-0.5
        attn_weight = query @ key.transpose(-2, -1) * scale
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value


def exists(val) -> bool:
    return val is not None


def default(val, d) -> Tensor:
    return val if exists(val) else d


def broadcat(tensors, dim=-1) -> Tensor:
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


def rotate_half(x) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[
        -1], f"Feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    out = torch.cat([t_left, t_transformed, t_right], dim=-1)
    return out.type(dtype)


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")

    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


class RotaryEmbedding(Module):
    def __init__(
            self,
            dim,
            custom_freqs: Union[Tensor, None] = None,
            freqs_for: Literal["lang", "pixel", "constant"] = "lang",
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False,
            use_xpos=False,
            xpos_scale_base=512,
            interpolate_factor=1.0,
            theta_rescale_factor=1.0,
            seq_before_head_dim=False,
            cache_if_possible=True,
            cache_max_seq_len=8192
    ) -> None:
        super(RotaryEmbedding, self).__init__()
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "spacetime":
            time_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

        if freqs_for == "spacetime":
            self.time_freqs = nn.Parameter(time_freqs, requires_grad=learned_freq)
        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer("cached_freqs", torch.zeros(cache_max_seq_len, dim), persistent=False)
        self.register_buffer("cached_freqs_seq_len", torch.tensor(0), persistent=False)

        self.learned_freq = learned_freq

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        self.use_xpos = use_xpos
        if use_xpos:
            scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
            self.scale_base = xpos_scale_base

            self.register_buffer("scale", scale, persistent=False)
            self.register_buffer("cached_scales", torch.zeros(cache_max_seq_len, dim), persistent=False)
            self.register_buffer("cached_scales_seq_len", torch.tensor(0), persistent=False)

            self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, freqs, seq_dim=None, offset=0, scale=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(
            scale), ("You must use '.rotate_queries_and_keys' method instead and pass in both queries and keys "
                     "for length extrapolatable rotary embeddings")

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        seq_freqs = self.forward(seq, freqs, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            seq_freqs = rearrange(seq_freqs, "n d -> n 1 d")
        return apply_rotary_emb(seq_freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim)

    def rotate_queries_and_keys(self, q, k, freqs, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        seq_freqs = self.forward(seq, freqs, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            seq_freqs = rearrange(seq_freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(seq_freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(seq_freqs, k, scale=scale ** -1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)
        return rotated_q, rotated_k

    def get_scale(self, t: Tensor, seq_len: Union[int, None] = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and exists(seq_len) and (offset + seq_len) <= self.cache_max_seq_len

        if should_cache and exists(self.cached_scales) and (seq_len + offset) <= self.cached_scales_seq_len.item():
            return self.cached_scales[offset: (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = repeat(scale, "n d -> n (d r)", r=2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len.copy_(seq_len)
        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            # Only allow pixel freqs for last two dimensions
            use_pixel = (self.freqs_for == "pixel" or self.freqs_for == "spacetime") and ind >= len(dims) - 2
            if use_pixel:
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            if self.freqs_for == "spacetime" and not use_pixel:
                seq_freqs = self.forward(pos, self.time_freqs, seq_len=dim)
            else:
                seq_freqs = self.forward(pos, self.freqs, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(seq_freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, freqs: Tensor, seq_len=None, offset=0):
        should_cache = self.cache_if_possible and not self.learned_freq and exists(
            seq_len) and self.freqs_for != "pixel" and (offset + seq_len) <= self.cache_max_seq_len

        if should_cache and exists(self.cached_freqs) and (offset + seq_len) <= self.cached_freqs_seq_len.item():
            return self.cached_freqs[offset: (offset + seq_len)].detach()
        else:
            freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
            freqs = repeat(freqs, "... n -> ... (n r)", r=2)

            if should_cache and offset == 0:
                self.cached_freqs[:seq_len] = freqs.detach()
                self.cached_freqs_seq_len.copy_(seq_len)
            return freqs

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[:x.shape[2]]


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, rot_emb: bool = False) -> None:
        super(SelfAttention, self).__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )

        self.rot_emb = rot_emb
        if rot_emb:
            self.rotary_embedding = RotaryEmbedding(dim=inner_dim)


    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q, self.rotary_embedding.freqs)
            k = self.rotary_embedding.rotate_queries_or_keys(k, self.rotary_embedding.freqs)
            q, k = map(lambda t: t.contiguous(), (q, k))
        out = scaled_dot_product_attention(q, k, v, is_causal=is_causal,scale=self.scale)
        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor) -> Tensor:
        t_len = x.shape[1]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, num_heads: int, dropout: float = 0.0, rot_emb: bool = False) -> None:
        super(CrossAttention, self).__init__()
        inner_dim = key_dim // num_heads
        value_dim = key_dim
        self.scale = inner_dim ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, key_dim, bias=False)
        self.to_k = nn.Linear(key_dim, key_dim, bias=False)
        self.to_v = nn.Linear(value_dim, key_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(key_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.rot_emb = rot_emb
        if rot_emb:
            self.rotary_embedding = RotaryEmbedding(dim=inner_dim)


    def forward(self, x: Tensor, kv: Tensor, attn_mask=None) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(kv)
        v = self.to_v(kv)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q, self.rotary_embedding.freqs)
            k = self.rotary_embedding.rotate_queries_or_keys(k, self.rotary_embedding.freqs)
            q, k = map(lambda t: t.contiguous(), (q, k))
        if attn_mask is not None:
            attn_mask = attn_mask[None, None, :, :]
            attn_mask = attn_mask.expand(q.shape[0], q.shape[1], -1, -1).contiguous()
        out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,scale=self.scale)
        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossBlock(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(CrossBlock, self).__init__()
        self.cross_attn = CrossAttention(query_dim, key_dim,  num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim)
        )

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(key_dim)

    def forward(self, x: Tensor, kv: Tensor, action: Tensor=None,
                attn_mask=None) -> Tensor:
        """
        x: Tensor[B, T, S, D], S is the number of spatial tokens, D is the feature dimension
        kv: Tensor[B, T, S, C]
        action: Tensor[B, T, S, D], applied to ffn, is optional to support action conditioning
        """
        t_len = x.shape[1]

        x = rearrange(x, "b t s e -> b (t s) e")
        x_ = self.norm1(x)
        kv_ = rearrange(kv, "b t s c -> b (t s) c")
        kv_ = self.norm3(kv_)
        x_ = self.cross_attn(x_, kv_,attn_mask=attn_mask)
        x = x + x_
        x = rearrange(x, "b (t s) e -> b t s e", t=t_len)

        # Feedforward
        x_ = self.norm2(x)
        if action is not None:
            x_ = self.ffn(x_ + action)
        else:
            x_ = self.ffn(x_)
        x = x + x_
        return x
    

class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_channel: int = -1, dropout: float = 0.0, init_values: float = 1e-5) -> None:
        super(SpatioTemporalBlock, self).__init__()
        self.num_channel = num_channel
        if num_channel > 1:
            self.norm1 = nn.ModuleList(
                [
                    nn.LayerNorm(model_dim)
                    for _ in range(num_channel)
                ]
            )
            self.spatial_attn = nn.ModuleList(
                [
                    SelfAttention(model_dim, num_heads, dropout=dropout)
                    for _ in range(num_channel)
                ]
            )
            self.ls1 = nn.ModuleList(
                [
                    LayerScale(model_dim, init_values=init_values)
                    for _ in range(num_channel)
                ]
            )
        else:
            self.norm1 = nn.LayerNorm(model_dim)
            self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
            self.ls1 = LayerScale(model_dim, init_values=init_values)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout, rot_emb=True)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        
        
        self.ls2 = LayerScale(model_dim, init_values=init_values)
        self.ls3 = LayerScale(model_dim, init_values=init_values)

    def forward(self, x: Tensor, causal_temporal: bool = False) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        if self.num_channel > 1:
            x_ = []
            for c in range(self.num_channel):
                x_c = x[:, c, :, :]  # (B, S, E)
                x_c_ = self.norm1[c](x_c)
                x_c_ = self.spatial_attn[c](x_c_)
                x_c = x_c + self.ls1[c](x_c_)  # weighted residual
                x_.append(x_c)
            x = torch.stack(x_, dim=1)  # (B, T, S, E)
        else:
            x = rearrange(x, "b t s e -> (b t) s e")
            x_ = self.norm1(x)
            x_ = self.spatial_attn(x_)
            x = x + self.ls1(x_) # weighted residual
            x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Temporal attention
        x = rearrange(x, "b t s e -> (b s) t e")
        x_ = self.norm2(x)
        if causal_temporal:
            x_ = self.temporal_attn(x_, is_causal=True)
        else:
            x_ = self.temporal_attn(x_)
        x = x + self.ls2(x_) # weighted residual
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)

        # Feedforward
        x_ = self.norm3(x)
        x_ = self.ffn(x_)
        x = x + self.ls3(x_) # weighted residual
        return x


class LayerScale(nn.Module):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239
    """

    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        """Initialize LayerScale module.

        Args:
            dim: Dimension.
            init_values: Initial value for scaling.
            inplace: If True, perform inplace operations.
        """
        super().__init__()
        self.inplace = inplace
        self.enable_scale = True if init_values > 0 else False
        if self.enable_scale:
            self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling."""
        if not self.enable_scale:
            return x
        return x.mul_(self.gamma) if self.inplace else x * self.gamma