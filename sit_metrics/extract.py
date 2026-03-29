"""
Register hooks on SiT blocks to capture:
- H^(ℓ): block output [B, N, D]
- x_attn^(ℓ): input to self-attn (for computing A^(ℓ) via attention_probs_from_timm)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sit_metrics.attention import attention_probs_from_timm, disable_fused_attention


@dataclass
class HookedActivations:
    """Per-forward capture for requested layers."""

    H: Dict[int, torch.Tensor] = field(default_factory=dict)
    attn_in: Dict[int, torch.Tensor] = field(default_factory=dict)
    A: Dict[int, torch.Tensor] = field(default_factory=dict)


def _make_attn_pre_hook(storage: Dict[int, torch.Tensor], layer_idx: int):
    def _hook(_module, inputs: Tuple[torch.Tensor, ...]):
        storage[layer_idx] = inputs[0].detach()

    return _hook


def _make_block_post_hook(storage: Dict[int, torch.Tensor], layer_idx: int):
    def _hook(_module, _inp, out: torch.Tensor):
        storage[layer_idx] = out.detach()

    return _hook


def register_metric_hooks(
    model: torch.nn.Module,
    layer_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[torch.utils.hooks.RemovableHandle], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    If layer_indices is None, hooks all layers [0 .. L-1].
    Returns handles, H_storage, attn_in_storage (empty until forward).
    """
    blocks = model.blocks
    L = len(blocks)
    if layer_indices is None:
        layer_indices = list(range(L))
    else:
        layer_indices = [int(i) for i in layer_indices]

    H_storage: Dict[int, torch.Tensor] = {}
    attn_storage: Dict[int, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for li in layer_indices:
        if li < 0 or li >= L:
            raise ValueError(f"layer index {li} out of range [0, {L})")
        blk = blocks[li]
        handles.append(blk.attn.register_forward_pre_hook(_make_attn_pre_hook(attn_storage, li)))
        handles.append(blk.register_forward_hook(_make_block_post_hook(H_storage, li)))

    return handles, H_storage, attn_storage


def compute_attention_probs(
    model: torch.nn.Module,
    attn_in: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Compute A^(ℓ) from stored attn inputs (call in no_grad)."""
    A: Dict[int, torch.Tensor] = {}
    for li, x_in in attn_in.items():
        blk = model.blocks[li]
        A[li] = attention_probs_from_timm(blk.attn, x_in)
    return A


def grid_size_from_N(N: int) -> Tuple[int, int]:
    import math

    g = int(math.sqrt(N))
    if g * g != N:
        raise ValueError(f"N={N} is not a square grid (expected p*q latent tokens)")
    return g, g


def forward_sit(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
    return model(x, t, y)


def forward_repa(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
    out = model(x, t, y)
    return out[0] if isinstance(out, tuple) else out


def run_forward_collect(
    model: torch.nn.Module,
    backend: str,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    layer_indices: Optional[Sequence[int]],
    compute_A: bool,
    sync_attn_path: bool = True,
) -> HookedActivations:
    """
    Single forward; fills H and optionally A for listed layers.
    If sync_attn_path, sets fused_attn=False so block forward matches manual A from attention_probs_from_timm.
    """
    if sync_attn_path:
        disable_fused_attention(model)
    handles, H_s, attn_s = register_metric_hooks(model, layer_indices)
    out = HookedActivations()
    try:
        if backend == "sit":
            forward_sit(model, x, t, y)
        else:
            forward_repa(model, x, t, y)
        out.H = dict(H_s)
        out.attn_in = dict(attn_s)
        if compute_A and attn_s:
            out.A = compute_attention_probs(model, attn_s)
    finally:
        for h in handles:
            h.remove()
    return out
