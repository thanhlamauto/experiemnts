"""
Self-attention probability matrix from timm.layers.Attention (matches non-fused path).
Output shape: [B, num_heads, N, N] (row = query i, col = key j).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from timm.layers.attention import maybe_add_mask, resolve_self_attn_mask


@torch.no_grad()
def attention_probs_from_timm(
    attn_module: torch.nn.Module,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    x: input to Attention forward (same tensor as modulate(norm1(...)) in SiTBlock).
    Returns softmax(QK^T/sqrt(d)) before dropout matmul V.
    """
    B, N, _C = x.shape
    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(
        2, 0, 3, 1, 4
    )
    q, k, v = qkv.unbind(0)
    q, k = attn_module.q_norm(q), attn_module.k_norm(k)
    q = q * attn_module.scale
    attn = q @ k.transpose(-2, -1)
    attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
    attn = maybe_add_mask(attn, attn_bias)
    attn = attn.softmax(dim=-1)
    return attn


def disable_fused_attention(model: torch.nn.Module) -> None:
    """Force timm Attention to expose weights via manual path (also used during metric forward)."""
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
