# Copyright 2025 MetalLiger Contributors
# Licensed under the Apache License 2.0
"""
MetalLiger — Fused Metal kernels for PyTorch MPS training on Apple Silicon.

Brings Liger Kernel-style operator fusion to Apple's Metal GPU backend,
eliminating intermediate tensor allocations and reducing Metal dispatch overhead
by 50%+ for transformer training.

Phases:
  Phase 3 (active):  Python autograd.Function fused ops (RMSNorm, SwiGLU, RoPE, CE)
  Phase 4a (active): torch.compile graph capture (use_metalliger_compile=True)
  Phase 4b-e (build): Native C++ Metal dispatch + torch.library registration
                       Build: python metalliger/setup_ext.py build_ext --inplace

Usage:
    from metalliger import apply_metalliger_to_qwen3vl

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-VL-4B")
    model = apply_metalliger_to_qwen3vl(model)
    # Training proceeds as normal — all ops are fused under the hood
"""

__version__ = "0.3.0"

from metalliger.ops.fused_rms_norm import MetalLigerRMSNorm
from metalliger.ops.fused_swiglu import MetalLigerSwiGLU
from metalliger.ops.fused_rope import MetalLigerRoPE
from metalliger.ops.fused_linear_cross_entropy import MetalLigerFusedLinearCrossEntropy
from metalliger.ops.fused_lora import (
    apply_fused_lora_qkv,
    apply_fused_lora_mlp,
    patch_fused_lora,
)
from metalliger.patch import apply_metalliger_to_qwen3vl, apply_metalliger_to_qwen3_5
from metalliger.ops.flash_attn import apply_mps_flash_attention, is_mps_flash_attn_available

# Phase 4: torch.compile wrapper
from metalliger._compile import apply_compile

# DeepSpeed-inspired: Contiguous gradient buffer
from metalliger.contiguous_grad_buffer import ContiguousGradBuffer

# Phase 4e: Register native ops with torch.library (no-op if C++ ext not built yet)
from metalliger._torch_library import register_metalliger_ops
register_metalliger_ops()


def _disable_transformers_causal_conv1d():
    """Unsloth mechanism: dynamically blocks Hugging Face from importing
    broken or missing causal_conv1d and flash-linear-attention kernels
    when running on non-CUDA platforms (like Apple Silicon MPS)."""
    try:
        from transformers.utils import import_utils as tf_import_utils
        
        # Block causal_conv1d
        if hasattr(tf_import_utils, "is_causal_conv1d_available"):
            tf_import_utils.is_causal_conv1d_available = lambda: False
        for attr in ["_causal_conv1d_available", "_is_causal_conv1d_available"]:
            if hasattr(tf_import_utils, attr):
                setattr(tf_import_utils, attr, False)
                
        # Block flash-linear-attention (if checked by HF)
        if hasattr(tf_import_utils, "is_flash_linear_attention_available"):
            tf_import_utils.is_flash_linear_attention_available = lambda: False
            
    except ImportError:
        pass

# Apply blocker immediately when MetalLiger is imported
_disable_transformers_causal_conv1d()

__all__ = [
    # Phase 3: Python fused ops
    "MetalLigerRMSNorm",
    "MetalLigerSwiGLU",
    "MetalLigerRoPE",
    "MetalLigerFusedLinearCrossEntropy",
    "apply_metalliger_to_qwen3vl",
    "apply_metalliger_to_qwen3_5",
    # Phase 3.5: Fused LoRA QKV + MLP backward
    "apply_fused_lora_qkv",
    "apply_fused_lora_mlp",
    "patch_fused_lora",
    # Phase 4a
    "apply_compile",
    # Phase 4e
    "register_metalliger_ops",
    # Phase 6: MPS Flash Attention
    "apply_mps_flash_attention",
    "is_mps_flash_attn_available",
    # DeepSpeed-inspired: Contiguous Gradient Buffer
    "ContiguousGradBuffer",
]
