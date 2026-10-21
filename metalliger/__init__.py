# Copyright 2025-2026 MetalLiger Contributors
# Licensed under the Apache License 2.0
"""
MetalLiger — Fused Metal kernels for PyTorch MPS training & inference on Apple Silicon.

Brings Liger Kernel-style operator fusion to Apple's Metal GPU backend,
eliminating intermediate tensor allocations and reducing Metal dispatch overhead
by 50%+ for transformer training and autoregressive VLM inference.

Usage (Inference):
    from metalliger import InferenceEngine
    from PIL import Image

    engine = InferenceEngine(model_id="Qwen/Qwen3-VL-4B")
    response = engine.generate(text_prompt="Explain this scan", images=[Image.open("scan.png")])

Usage (Training / Monkey-patching):
    from metalliger import apply_metalliger_to_qwen3vl

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-VL-4B")
    model = apply_metalliger_to_qwen3vl(model)
"""

__version__ = "0.5.0"

try:
    from .ops.fused_rms_norm import MetalLigerRMSNorm
    from .ops.fused_swiglu import MetalLigerSwiGLU
    from .ops.fused_rope import MetalLigerRoPE
    from .ops.fused_linear_cross_entropy import MetalLigerFusedLinearCrossEntropy
    from .ops.fused_lora import (
        apply_fused_lora_qkv,
        apply_fused_lora_mlp,
        patch_fused_lora,
    )
    from .patch import apply_metalliger_to_qwen3vl, apply_metalliger_to_qwen3_5
    from .ops.flash_attn import apply_mps_flash_attention, is_mps_flash_attn_available
    from ._compile import apply_compile
    from .contiguous_grad_buffer import ContiguousGradBuffer
    from .inference import InferenceEngine
    from .generation import M4StaticCache
    from ._torch_library import register_metalliger_ops
except ImportError:
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
    from metalliger._compile import apply_compile
    from metalliger.contiguous_grad_buffer import ContiguousGradBuffer
    from metalliger.inference import InferenceEngine
    from metalliger.generation import M4StaticCache
    from metalliger._torch_library import register_metalliger_ops

# Phase 4e: Register native ops with torch.library (no-op if C++ ext not built yet)
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
    "__version__",
    # Inference
    "InferenceEngine",
    "M4StaticCache",
    # Fused ops
    "MetalLigerRMSNorm",
    "MetalLigerSwiGLU",
    "MetalLigerRoPE",
    "MetalLigerFusedLinearCrossEntropy",
    "apply_metalliger_to_qwen3vl",
    "apply_metalliger_to_qwen3_5",
    # Fused LoRA QKV + MLP backward
    "apply_fused_lora_qkv",
    "apply_fused_lora_mlp",
    "patch_fused_lora",
    # Compilation
    "apply_compile",
    # Torch library registration
    "register_metalliger_ops",
    # MPS Flash Attention
    "apply_mps_flash_attention",
    "is_mps_flash_attn_available",
    # Contiguous Gradient Buffer
    "ContiguousGradBuffer",
]
