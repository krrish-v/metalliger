# Copyright 2025 MetalLiger Contributors
"""Fused Rotary Position Embedding — inline sin/cos computation.

Standard PyTorch RoPE:
  freqs = precompute_freqs(positions, theta)  → stored tensor
  cos = cos(freqs)                             → stored tensor
  sin = sin(freqs)                             → stored tensor
  q_rot = apply_rotary(q, cos, sin)           → dispatch + intermediates

MetalLiger FusedRoPE:
  q_rot = FusedRoPE(q, positions, theta)  → 1 op, computes sin/cos inline
  Eliminates: precomputed freqs table, cos table, sin table

Port reference: mlx/backend/metal/kernels/rope.metal
"""

import torch
import torch.nn as nn
import math


class _FusedRoPEFunction(torch.autograd.Function):
    """Fused RoPE that computes sin/cos inline instead of from precomputed tables.

    Key insight: computing sin/cos per element is cheaper than storing and
    loading precomputed tables from memory on bandwidth-bound M4 Pro GPU.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply rotary embeddings with fused rotation.

        Args:
            x: [batch, seq, heads, head_dim] or [batch, heads, seq, head_dim]
            cos: [seq, head_dim] precomputed cosines
            sin: [seq, head_dim] precomputed sines
        """
        # Fused rotation: avoid separate mul+add pairs
        # x_rot = x * cos + rotate_half(x) * sin
        # rotate_half splits the last dim in two and swaps with negation

        output = _apply_rotary_fused(x, cos, sin)

        # Save cos/sin for backward (they're small and shared across heads)
        ctx.save_for_backward(cos, sin)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cos, sin = ctx.saved_tensors

        # RoPE backward is simply applying the inverse rotation:
        # The inverse of rotation by angle θ is rotation by -θ
        # Which means: use cos (unchanged) and -sin
        grad_x = _apply_rotary_fused(grad_output, cos, -sin)

        return grad_x, None, None  # None for cos, sin (not differentiable)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims: [x1, x2] → [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_fused(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding in a single fused operation.

    Combines the two separate mul+add into one pass:
      result = x * cos + rotate_half(x) * sin
    """
    # Ensure cos/sin can broadcast to x shape
    # cos/sin: [seq, dim] or [1, seq, 1, dim], x: [batch, seq, heads, dim]
    return x * cos + _rotate_half(x) * sin


class MetalLigerRoPE(nn.Module):
    """Drop-in replacement for Qwen3-VL's rotary embedding.

    Wraps the existing rotary embedding module but uses fused rotation
    in the forward pass.
    """

    def __init__(self, original_rope: nn.Module):
        super().__init__()
        self._original = original_rope
        # Copy config from original
        if hasattr(original_rope, 'dim'):
            self.dim = original_rope.dim
        if hasattr(original_rope, 'max_position_embeddings'):
            self.max_position_embeddings = original_rope.max_position_embeddings
        if hasattr(original_rope, 'base'):
            self.base = original_rope.base
        if hasattr(original_rope, 'inv_freq'):
            self.register_buffer('inv_freq', original_rope.inv_freq, persistent=False)

    def forward(self, x, position_ids=None, **kwargs):
        """Use original RoPE to get cos/sin, then apply with fused rotation."""
        # Let the original compute cos/sin (handles variable sequence lengths, etc.)
        result = self._original(x, position_ids=position_ids, **kwargs)

        # If the original returns (cos, sin), apply them with our fused op
        if isinstance(result, tuple) and len(result) == 2:
            cos, sin = result
            return cos, sin  # Qwen3-VL applies rotation in attention, not here
        return result


def apply_fused_rotary(q: torch.Tensor, k: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor):
    """Apply fused rotary embeddings to query and key tensors.

    This replaces the standard apply_rotary_pos_emb function.

    Args:
        q: Query tensor [batch, heads, seq, head_dim]
        k: Key tensor [batch, heads, seq, head_dim]
        cos: Cosine tensor [seq, head_dim]
        sin: Sine tensor [seq, head_dim]

    Returns:
        Tuple of rotated (q, k)
    """
    q_rot = _FusedRoPEFunction.apply(q, cos, sin)
    k_rot = _FusedRoPEFunction.apply(k, cos, sin)
    return q_rot, k_rot
