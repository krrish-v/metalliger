# Copyright 2025 MetalLiger Contributors
"""Fused DeltaNet Operations for Qwen3.5.

Qwen3.5 hybrid models rely heavily on Gated DeltaNet (Linear Attention) layers.
On CUDA, these use specialized flash-linear-attention and causal-conv1d kernels.
On MPS (Apple Silicon), PyTorch falls back to generic operations.

CRITICAL PRECISION ISSUE:
If these operations run in bfloat16 (the default mixed-precision type), the
recurrent state accumulation (prefix-scan) across thousands of tokens suffers
catastrophic rounding errors due to bfloat16's limited 7-bit mantissa. This
causes severe logit drift and collapses the model's reasoning capabilities.

SOLUTION:
Wrap HuggingFace's own correct implementation and force it to execute in
float32 by (a) upcasting the input and (b) disabling torch.autocast inside
the layer. PyTorch's type promotion (float32 input × bfloat16 weight → float32)
ensures all internal accumulations stay in float32.

GRADIENT CHECKPOINTING COMPATIBILITY:
We MUST NOT temporarily swap parameter dtypes, because gradient checkpointing
replays the forward pass during backward — if params are in a different state
during recomputation, the intermediate tensor dtypes mismatch and PyTorch raises
a CheckpointError. Instead, we only upcast the INPUT and disable autocast,
which is deterministic across forward and recomputation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)



class UpcastConv1d(nn.Module):
    """Wraps nn.Conv1d to dynamically cast its weights to match the input dtype.
    
    MPS F.conv1d is strictly typed and crashes if input is float32 but weight
    is bfloat16. This wrapper ensures the weight matches the input dtype dynamically
    during the forward pass, which preserves gradient checkpointing compatibility
    (no in-place parameter mutation).
    """
    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        self.conv = conv
        
    @property
    def weight(self):
        return self.conv.weight
        
    @property
    def bias(self):
        return self.conv.bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv.weight.to(x.dtype)
        bias = self.conv.bias.to(x.dtype) if self.conv.bias is not None else None
        return F.conv1d(
            x, weight, bias,
            self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
        )


def _make_fused_deltanet_forward(layer):
    """Creates a precision-hardened forward pass for Qwen3_5GatedDeltaNet.

    Strategy:
    1. Disable torch.autocast inside this layer (prevents bfloat16 downcasting)
    2. Upcast hidden_states input to float32
    3. Call HF's original forward — PyTorch type promotion ensures all matmuls
       (float32 input × bfloat16 weight) produce float32 outputs
    4. Downcast output back to original dtype

    This is gradient-checkpointing safe because it produces identical dtypes
    during both forward and recomputation (no parameter mutation).
    """
    # Save the original forward from the class (unbound method)
    original_forward = layer.__class__.forward

    def forward(hidden_states: torch.Tensor, *args, **kwargs):
        original_dtype = hidden_states.dtype

        # Skip if already float32 (no precision issue)
        if original_dtype == torch.float32:
            return original_forward(layer, hidden_states, *args, **kwargs)

        # ── Float32 Precision Enforcement ──
        # Disable autocast so PyTorch doesn't silently downcast our float32
        # inputs back to bfloat16. With autocast disabled:
        #   F.linear(float32_input, bfloat16_weight) → float32 output
        #   F.conv1d(float32_input, bfloat16_weight) → float32 output
        # This ensures the prefix-scan accumulates in float32.
        hidden_states_f32 = hidden_states.float()

        with torch.autocast(device_type="cpu", enabled=False), \
             torch.autocast(device_type="mps", enabled=False):
            result = original_forward(layer, hidden_states_f32, *args, **kwargs)

        # Downcast output back to original dtype
        if isinstance(result, tuple):
            out_list = list(result)
            if out_list[0] is not None and isinstance(out_list[0], torch.Tensor):
                out_list[0] = out_list[0].to(original_dtype)
            return tuple(out_list)
        elif isinstance(result, torch.Tensor):
            return result.to(original_dtype)
        else:
            return result

    return forward


def patch_deltanet(model: nn.Module) -> int:
    """Patches all GatedDeltaNet layers for numerical stability.

    Wraps HuggingFace's own DeltaNet forward with float32 precision enforcement.
    Compatible with gradient checkpointing and torch.autocast.
    Returns the number of patched modules.
    """
    count = 0

    def _patch_recursive(module: nn.Module):
        nonlocal count
        for name, child in module.named_children():
            if type(child).__name__ == "Qwen3_5GatedDeltaNet":
                # Fix MPS strict typing for conv1d
                if hasattr(child, "conv1d") and not isinstance(child.conv1d, UpcastConv1d):
                    child.conv1d = UpcastConv1d(child.conv1d)
                    
                child.forward = _make_fused_deltanet_forward(child)
                count += 1
                logger.debug(f"MetalLiger: patched {name} → float32 DeltaNet")

            _patch_recursive(child)

    _patch_recursive(model)
    return count
