"""
Fused Metal operations and attention kernels for MetalLiger on Apple Silicon.
"""

try:
    from .fused_rms_norm import MetalLigerRMSNorm
    from .fused_swiglu import MetalLigerSwiGLU
    from .fused_rope import MetalLigerRoPE
    from .fused_linear_cross_entropy import MetalLigerFusedLinearCrossEntropy
    from .flash_attn import apply_mps_flash_attention, is_mps_flash_attn_available
    from .fused_deltanet import patch_deltanet, UpcastConv1d
    from .fused_lora import (
        apply_fused_lora_qkv,
        apply_fused_lora_mlp,
        patch_fused_lora,
    )
except ImportError:
    from metalliger.ops.fused_rms_norm import MetalLigerRMSNorm
    from metalliger.ops.fused_swiglu import MetalLigerSwiGLU
    from metalliger.ops.fused_rope import MetalLigerRoPE
    from metalliger.ops.fused_linear_cross_entropy import MetalLigerFusedLinearCrossEntropy
    from metalliger.ops.flash_attn import apply_mps_flash_attention, is_mps_flash_attn_available
    from metalliger.ops.fused_deltanet import patch_deltanet, UpcastConv1d
    from metalliger.ops.fused_lora import (
        apply_fused_lora_qkv,
        apply_fused_lora_mlp,
        patch_fused_lora,
    )

__all__ = [
    "MetalLigerRMSNorm",
    "MetalLigerSwiGLU",
    "MetalLigerRoPE",
    "MetalLigerFusedLinearCrossEntropy",
    "apply_mps_flash_attention",
    "is_mps_flash_attn_available",
    "patch_deltanet",
    "UpcastConv1d",
    "apply_fused_lora_qkv",
    "apply_fused_lora_mlp",
    "patch_fused_lora",
]
