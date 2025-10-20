"""
Low-Rank Adaptation (LoRA) modules for efficient fine-tuning of linear layers.

Paper: https://arxiv.org/abs/2106.09685

Formulation: h = W x + ΔW x = W x + s B A x.
Here, W is the original weight matrix, and the LoRA adaptation ΔW is factorized into two low-rank matrices A and B, with scaling factor s = lora_alpha / r.
--------------------------------------------------------------
Serial LoRA introduces a shared low-rank matrix serially composite with
the attention mechanism.
Paper: https://arxiv.org/pdf/2503.17750
"""
import re
import torch
import math
from typing import Iterable, Optional

import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a nn.Linear with a LoRA adapter (B @ A) added to the forward path."""

    def __init__(self, wrapped: nn.Linear, r: int = 4, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        super().__init__()
        self.wrapped = wrapped
        in_features = wrapped.in_features
        out_features = wrapped.out_features

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if r > 0:
            # A: in -> r, B: r -> out
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            # init
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            # dummy placeholders for convenience
            self.lora_A = None
            self.lora_B = None

        # By default, freeze original weight when using LoRA adapters (common recipe)
        self.wrapped.weight.requires_grad = False
        if self.wrapped.bias is not None:
            self.wrapped.bias.requires_grad = False

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.wrapped(x)
        if self.r > 0 and not self.merged:
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return base + lora_out
        return base

    def merge(self):
        """Merge LoRA weights into the wrapped linear weight (adds B @ A to W)."""
        if self.r <= 0 or self.merged:
            return
        # compute delta = (B @ A) * scaling
        # B.weight: (out, r), A.weight: (r, in) -> B @ A has shape (out, in)
        delta = torch.matmul(self.lora_B.weight, self.lora_A.weight) * self.scaling
        self.wrapped.weight.data += delta
        self.merged = True
        # after merge, we can free LoRA grads
        self.lora_A.weight.requires_grad = False
        self.lora_B.weight.requires_grad = False

    def unmerge(self):
        """Remove previously merged LoRA weights from the wrapped linear weight."""
        if self.r <= 0 or not self.merged:
            return
        delta = torch.matmul(self.lora_B.weight, self.lora_A.weight) * self.scaling
        self.wrapped.weight.data -= delta
        self.merged = False
        self.lora_A.weight.requires_grad = True
        self.lora_B.weight.requires_grad = True


def _get_parent_module(root: nn.Module, module_name: str):
    """Return (parent_module, attr_name) for a dotted module_name under root."""
    parts = module_name.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora(
    model: nn.Module,
    target_modules: str = "mlp",
    r: int = 4,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
) -> int:
    """
    Replace matching nn.Linear modules in `model` with LoRALinear wrappers.
    target_modules: iterable of substrings; a Linear is wrapped if any substring is in its full name.
    Returns number of wrapped modules.
    """
    pattern = re.compile(target_modules)

    lora_modules = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if pattern.search(name):
            parent, attr = _get_parent_module(model, name)
            orig = getattr(parent, attr)
            setattr(parent, attr, LoRALinear(orig, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
            lora_modules.append(name)
    return lora_modules


def set_lora_trainable(model: nn.Module, train_lora_only: bool = True):
    """
    Set requires_grad so that only LoRA adapter parameters are trainable.
    If train_lora_only is False, leaves all params' requires_grad unchanged.
    """
    if not train_lora_only:
        return
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if m.lora_A is not None:
                m.lora_A.weight.requires_grad = True
                m.lora_B.weight.requires_grad = True
            if m.wrapped.bias is not None:
                # typically keep original bias frozen; if desired, unfreeze here
                pass


def merge_all_lora(model: nn.Module):
    """Merge all LoRA modules into their wrapped linear modules."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_all_lora(model: nn.Module):
    """Unmerge all LoRA modules (revert merged weights)."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()
