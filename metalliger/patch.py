# Copyright 2025 MetalLiger Contributors
"""Monkey-patching system for HuggingFace models.

Replaces standard model operations with MetalLiger fused equivalents.
Designed for minimal API surface — one function call to fuse everything.

Supported architectures:
  - Qwen3-VL  (pure transformer + vision):  apply_metalliger_to_qwen3vl()
  - Qwen3.5   (hybrid SSM+Attention + vision): apply_metalliger_to_qwen3_5()

Qwen3.5 ARCHITECTURE NOTES:
  The language model alternates between two layer types in a 3:1 pattern:
    - Qwen3_5GatedDeltaNet  (linear SSM attention) — layers 0-2, 4-6, 8-10…
    - Qwen3_5Attention       (standard GQA)          — layers 3, 7, 11, 15…

  SAFE to fuse:
    ✅ input_layernorm / post_attention_layernorm  (plain Qwen3_5RMSNorm)
    ✅ Qwen3_5MLP with gate_proj/up_proj/down_proj  (SwiGLU, all 24 layers)
    ✅ Final language model norm (Qwen3_5RMSNorm)

  MUST NOT fuse:
    ❌ Qwen3_5RMSNormGated inside GatedDeltaNet.norm
       — uses a gated forward path with state: output = norm(x) * sigmoid(gate)
       — different math than plain RMSNorm; replacing it silently corrupts SSM state
    ❌ Vision encoder LayerNorm / GELU — not RMSNorm, different eps semantics
    ❌ q_norm / k_norm inside Qwen3_5Attention — small [256]-dim, not a perf target
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from metalliger.ops.fused_rms_norm import MetalLigerRMSNorm
from metalliger.ops.fused_swiglu import MetalLigerSwiGLU
from metalliger.ops.fused_lora import patch_fused_lora
from metalliger.ops.fused_deltanet import patch_deltanet
from metalliger.ops.flash_attn import apply_mps_flash_attention, is_mps_flash_attn_available
from metalliger.generation import patch_generator

logger = logging.getLogger(__name__)


def _find_layers(model: nn.Module):
    """Find transformer decoder layers regardless of model wrapping.

    Handles:
      - Bare model: model.model.layers (Qwen3VLForConditionalGeneration)
      - PEFT-wrapped: model.base_model.model.model.layers (PeftModel → LoraModel → Qwen3VL)
      - Other nesting: walks up to 5 levels deep looking for a 'layers' attribute
    """
    # All possible paths to transformer layers, ordered most-specific first
    candidates = [
        # PEFT + Qwen3-VL: peft_model.base_model.model.model.language_model.layers
        lambda m: getattr(getattr(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'model', None), 'language_model', None), 'layers', None),
        
        # PEFT wrapped (LoRA): PeftModel.base_model.model.model.layers
        lambda m: getattr(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'model', None), 'layers', None),
        
        # Qwen3-VL nested architecture: model.model.language_model.layers
        lambda m: getattr(getattr(getattr(m, 'model', None), 'language_model', None), 'layers', None),
        
        # Standard HF: model.model.layers
        lambda m: getattr(getattr(m, 'model', None), 'layers', None),
        
        # Direct: model.layers
        lambda m: getattr(m, 'layers', None),
        
        # Accelerate wrapped
        lambda m: getattr(getattr(getattr(m, 'module', None), 'model', None), 'layers', None),
    ]

    for getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue

    return None


def _find_final_norm(model: nn.Module):
    """Find the final RMSNorm (model.model.norm or deeper)."""
    candidates = [
        # PEFT + Qwen3-VL
        lambda m: getattr(getattr(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'model', None), 'language_model', None), 'norm', None),
        
        # Qwen3-VL nested architecture
        lambda m: getattr(getattr(getattr(m, 'model', None), 'language_model', None), 'norm', None),
        
        # Standard PEFT
        lambda m: getattr(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'model', None), 'norm', None),
        
        # Standard HF
        lambda m: getattr(getattr(m, 'model', None), 'norm', None),
    ]
    for getter in candidates:
        try:
            norm = getter(model)
            if norm is not None and hasattr(norm, 'weight'):
                return norm, getter
        except (AttributeError, TypeError):
            continue
    return None, None


def _replace_rms_norms(model: nn.Module) -> int:
    """Replace all RMSNorm layers with MetalLiger fused versions.

    Returns the number of layers replaced.
    """
    count = 0

    layers = _find_layers(model)
    if layers is None:
        logger.warning("MetalLiger: could not find transformer layers for RMSNorm patching")
        return 0

    for layer in layers:
        # Input layernorm
        if hasattr(layer, 'input_layernorm'):
            old_norm = layer.input_layernorm
            eps = getattr(old_norm, 'variance_epsilon',
                          getattr(old_norm, 'eps', 1e-6))
            hidden_size = old_norm.weight.shape[0]
            new_norm = MetalLigerRMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                weight=old_norm.weight.data,
            )
            layer.input_layernorm = new_norm
            count += 1

        # Post-attention layernorm
        if hasattr(layer, 'post_attention_layernorm'):
            old_norm = layer.post_attention_layernorm
            eps = getattr(old_norm, 'variance_epsilon',
                          getattr(old_norm, 'eps', 1e-6))
            hidden_size = old_norm.weight.shape[0]
            new_norm = MetalLigerRMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                weight=old_norm.weight.data,
            )
            layer.post_attention_layernorm = new_norm
            count += 1

    # Final norm (handles PEFT wrapping)
    old_norm, _ = _find_final_norm(model)
    if old_norm is not None:
        eps = getattr(old_norm, 'variance_epsilon',
                      getattr(old_norm, 'eps', 1e-6))
        hidden_size = old_norm.weight.shape[0]
        new_norm = MetalLigerRMSNorm(
            hidden_size=hidden_size,
            eps=eps,
            weight=old_norm.weight.data,
        )
        # Set the norm on the correct parent — use safe getattr chains
        try:
            # PEFT + VLM: model.base_model.model.model.language_model.norm
            peft_lang = getattr(getattr(getattr(getattr(model, 'base_model', None), 'model', None), 'model', None), 'language_model', None)
            if peft_lang is not None and hasattr(peft_lang, 'norm'):
                peft_lang.norm = new_norm
                count += 1
            # PEFT: model.base_model.model.model.norm
            elif getattr(getattr(getattr(getattr(model, 'base_model', None), 'model', None), 'model', None), 'norm', None) is not None:
                model.base_model.model.model.norm = new_norm
                count += 1
            # VLM (no PEFT): model.model.language_model.norm
            elif getattr(getattr(getattr(model, 'model', None), 'language_model', None), 'norm', None) is not None:
                model.model.language_model.norm = new_norm
                count += 1
            # Standard HF: model.model.norm
            elif getattr(getattr(model, 'model', None), 'norm', None) is not None:
                model.model.norm = new_norm
                count += 1
        except AttributeError as e:
            logger.debug(f"MetalLiger: could not replace final norm. {e}")

    return count


def _replace_mlps(model: nn.Module) -> int:
    """Replace SwiGLU MLPs with MetalLiger fused versions.

    Returns the number of layers replaced.
    """
    count = 0
    layers = _find_layers(model)
    if layers is None:
        return 0

    for layer in layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
            original_mlp = layer.mlp
            layer.mlp = MetalLigerSwiGLU(original_mlp)
            count += 1

    return count


def apply_metalliger_to_qwen3vl(
    model: nn.Module,
    fuse_rms_norm: bool = True,
    fuse_swiglu: bool = True,
    fuse_rope: bool = False,
    fuse_cross_entropy: bool = False,
    fuse_lora: bool = False,  # DISABLED: dynamic lambda-shadowing of q/k/v_proj causes silent
                              # precision drift and logit erosion during extended training runs.
                              # Standard PEFT backward is numerically safer. Memory cost: +1.8GB.
    fuse_flash_attn: bool = False,  # Phase 6: MPS Flash Attention (disabled by default)
    use_optimized_generator: bool = False, # DISABLED: breaks GRPO sampling
) -> nn.Module:
    """Apply MetalLiger fused operations to a Qwen3-VL model.

    This is the main entry point. Call once after loading the model.

    Args:
        model: A Qwen3-VL model (or similar architecture)
        fuse_rms_norm: Replace RMSNorm with fused version (default: True)
        fuse_swiglu: Replace SwiGLU MLP with fused version (default: True)
        fuse_rope: Replace RoPE with fused version (default: False — experimental)
        fuse_cross_entropy: Whether to use fused CE (applied in trainer)

    Returns:
        The same model with fused operations applied in-place
    """
    stats = {}

    # Flash Attention is applied FIRST — patches F.scaled_dot_product_attention
    # globally so all subsequent attention ops (in both patched and unpatched
    # modules) automatically benefit from O(N) memory + 1.8x speedup.
    if fuse_flash_attn:
        success = apply_mps_flash_attention(bf16_backward=True)
        stats['flash_attn'] = 'active' if success else 'unavailable'

    if fuse_rms_norm:
        n = _replace_rms_norms(model)
        stats['rms_norm'] = n
        logger.info(f"MetalLiger: replaced {n} RMSNorm layers with fused versions")

    if fuse_swiglu:
        n = _replace_mlps(model)
        stats['swiglu'] = n
        logger.info(f"MetalLiger: replaced {n} MLP layers with fused SwiGLU")

    if fuse_lora:
        n = patch_fused_lora(model)
        stats['fused_lora'] = n
        logger.info(f"MetalLiger: fused LoRA QKV+MLP marked {n} modules (Phase 3.5)")
    else:
        logger.info(
            "MetalLiger: fused LoRA DISABLED (default). Using standard PEFT backward "
            "for numerical stability. Set fuse_lora=True to re-enable (saves ~1.8GB but "
            "risks precision drift on long runs)."
        )

    if use_optimized_generator:
        patch_generator(model)
        stats['generator'] = 'optimized'
        logger.info("MetalLiger: applied Phase 5 optimized MPS generator")

    if fuse_rope:
        logger.info("MetalLiger: fused RoPE is experimental — use apply_fused_rotary()")
        stats['rope'] = 'available'

    if fuse_cross_entropy:
        logger.info("MetalLiger: fused CE — use MetalLigerFusedLinearCrossEntropy in trainer")
        stats['cross_entropy'] = 'available'



    total = sum(v for v in stats.values() if isinstance(v, int))
    logger.info(
        f"MetalLiger: {total} total layers patched. "
        f"Details: {stats}"
    )

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Qwen3.5 Support (Hybrid SSM + Attention + Vision)
# ══════════════════════════════════════════════════════════════════════════════

def _find_language_layers_qwen3_5(model: nn.Module):
    """Find the language_model decoder layers for Qwen3.5.

    Qwen3.5 path: model.model.language_model.layers
    PEFT-wrapped:  model.base_model.model.model.language_model.layers
    """
    candidates = [
        lambda m: getattr(getattr(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'model', None), 'language_model', None), 'layers', None),
        lambda m: getattr(getattr(getattr(m, 'model', None), 'language_model', None), 'layers', None),
        lambda m: getattr(getattr(m, 'model', None), 'layers', None),
        lambda m: getattr(m, 'layers', None),
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    return None


def _find_final_norm_qwen3_5(model: nn.Module):
    """Find the final language model RMSNorm for Qwen3.5."""
    candidates = [
        lambda m: getattr(getattr(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'model', None), 'language_model', None), 'norm', None),
        lambda m: getattr(getattr(getattr(m, 'model', None), 'language_model', None), 'norm', None),
        lambda m: getattr(getattr(m, 'model', None), 'norm', None),
    ]
    for getter in candidates:
        try:
            norm = getter(model)
            if norm is not None and hasattr(norm, 'weight'):
                # Guard: must be a plain RMSNorm, NOT the gated variant
                cls_name = type(norm).__name__
                if 'Gated' not in cls_name:
                    return norm
        except (AttributeError, TypeError):
            continue
    return None


def _replace_rms_norms_qwen3_5(model: nn.Module) -> int:
    """Replace standard RMSNorm layers in Qwen3.5 language model.

    CRITICAL SAFETY RULE:
    Only replaces input_layernorm and post_attention_layernorm — both are
    plain Qwen3_5RMSNorm. The Qwen3_5RMSNormGated inside GatedDeltaNet.norm
    is SKIPPED because it uses gated forward math incompatible with our
    standard fused RMSNorm implementation.
    """
    count = 0
    layers = _find_language_layers_qwen3_5(model)
    if layers is None:
        logger.warning("MetalLiger[Qwen3.5]: could not find language model layers")
        return 0

    for layer_idx, layer in enumerate(layers):
        # input_layernorm — plain Qwen3_5RMSNorm in ALL layer types
        if hasattr(layer, 'input_layernorm'):
            old_norm = layer.input_layernorm
            cls_name = type(old_norm).__name__
            if 'Gated' not in cls_name:  # Safety guard
                eps = getattr(old_norm, 'variance_epsilon', getattr(old_norm, 'eps', 1e-6))
                new_norm = MetalLigerRMSNorm(
                    hidden_size=old_norm.weight.shape[0],
                    eps=eps,
                    weight=old_norm.weight.data,
                )
                layer.input_layernorm = new_norm
                count += 1
            else:
                logger.debug(f"MetalLiger[Qwen3.5]: layer {layer_idx} input_layernorm is Gated — skipped")

        # post_attention_layernorm — plain Qwen3_5RMSNorm in ALL layer types
        if hasattr(layer, 'post_attention_layernorm'):
            old_norm = layer.post_attention_layernorm
            cls_name = type(old_norm).__name__
            if 'Gated' not in cls_name:  # Safety guard
                eps = getattr(old_norm, 'variance_epsilon', getattr(old_norm, 'eps', 1e-6))
                new_norm = MetalLigerRMSNorm(
                    hidden_size=old_norm.weight.shape[0],
                    eps=eps,
                    weight=old_norm.weight.data,
                )
                layer.post_attention_layernorm = new_norm
                count += 1
            else:
                logger.debug(f"MetalLiger[Qwen3.5]: layer {layer_idx} post_attention_layernorm is Gated — skipped")

    # Final language model norm
    final_norm = _find_final_norm_qwen3_5(model)
    if final_norm is not None:
        eps = getattr(final_norm, 'variance_epsilon', getattr(final_norm, 'eps', 1e-6))
        new_norm = MetalLigerRMSNorm(
            hidden_size=final_norm.weight.shape[0],
            eps=eps,
            weight=final_norm.weight.data,
        )
        # Set on correct parent — use safe getattr chains to avoid crashing
        # when the model hierarchy doesn't match expectations (e.g. Qwen3.5-2B-Base
        # loaded via AutoModelForImageTextToText has Qwen3_5Model without .model)
        try:
            # PEFT + VLM: model.base_model.model.model.language_model.norm
            peft_lang = getattr(getattr(getattr(getattr(model, 'base_model', None), 'model', None), 'model', None), 'language_model', None)
            if peft_lang is not None and hasattr(peft_lang, 'norm'):
                peft_lang.norm = new_norm
                count += 1
            # PEFT + bare LLM: model.base_model.model.model.norm
            elif getattr(getattr(getattr(getattr(model, 'base_model', None), 'model', None), 'model', None), 'norm', None) is not None:
                model.base_model.model.model.norm = new_norm
                count += 1
            # VLM (no PEFT): model.model.language_model.norm
            elif getattr(getattr(getattr(model, 'model', None), 'language_model', None), 'norm', None) is not None:
                model.model.language_model.norm = new_norm
                count += 1
            # Bare LLM (no PEFT): model.model.norm
            elif getattr(getattr(model, 'model', None), 'norm', None) is not None:
                model.model.norm = new_norm
                count += 1
            else:
                logger.debug("MetalLiger[Qwen3.5]: could not find parent for final norm replacement")
            logger.debug("MetalLiger[Qwen3.5]: final language model norm replaced")
        except AttributeError as e:
            logger.debug(f"MetalLiger[Qwen3.5]: could not replace final norm: {e}")

    return count


def _replace_mlps_qwen3_5(model: nn.Module) -> int:
    """Replace Qwen3_5MLP with MetalLiger fused SwiGLU in all 24 layers.

    Both GatedDeltaNet layers and Attention layers share the same
    Qwen3_5MLP structure (gate_proj + up_proj + down_proj + SiLUActivation),
    so MetalLigerSwiGLU is a safe drop-in for all of them.
    """
    count = 0
    layers = _find_language_layers_qwen3_5(model)
    if layers is None:
        return 0

    for layer in layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
            layer.mlp = MetalLigerSwiGLU(layer.mlp)
            count += 1

    return count


def apply_metalliger_to_qwen3_5(
    model: nn.Module,
    fuse_rms_norm: bool = True,
    fuse_swiglu: bool = True,
    fuse_lora: bool = False,   # DISABLED: dynamic lambda-shadowing of q/k/v_proj causes silent
                               # precision drift and logit erosion during extended training runs.
                               # Standard PEFT backward is numerically safer. Memory cost: +1.8GB.
    fuse_flash_attn: bool = False,  # Phase 6: MPS Flash Attention (disabled by default)
) -> nn.Module:
    """Apply MetalLiger fused operations to a Qwen3.5 hybrid SSM+Attention model.

    Supports Qwen/Qwen3.5-2B (and larger variants with the same architecture).
    Safe to call on both bare models and PEFT-wrapped models.

    What is fused:
      - input_layernorm / post_attention_layernorm in all 24 decoder layers
        (plain Qwen3_5RMSNorm — NOT the Qwen3_5RMSNormGated inside DeltaNet)
      - Final language model norm
      - All 24 Qwen3_5MLP blocks (SwiGLU gate + up fused into one op)
      - LoRA QKV + MLP backward (if PEFT adapters present)

    What is NOT touched:
      - Qwen3_5RMSNormGated inside GatedDeltaNet (different gated math)
      - Vision encoder (LayerNorm + GELU — not RMSNorm)
      - q_norm / k_norm inside Attention layers (small dim, negligible)

    Args:
        model: A Qwen3.5 model (Qwen3_5ForConditionalGeneration or PEFT-wrapped)
        fuse_rms_norm: Replace standard RMSNorm layers (default: True)
        fuse_swiglu: Replace all Qwen3_5MLP blocks (default: True)
        fuse_lora: Activate fused LoRA QKV+MLP backward (default: False — disabled for stability)

    Returns:
        The same model with fused operations applied in-place.
    """
    stats = {}

    # Flash Attention patched FIRST (global F.scaled_dot_product_attention)
    if fuse_flash_attn:
        success = apply_mps_flash_attention(bf16_backward=True)
        stats['flash_attn'] = 'active' if success else 'unavailable'

    if fuse_rms_norm:
        n = _replace_rms_norms_qwen3_5(model)
        stats['rms_norm'] = n
        logger.info(
            f"MetalLiger[Qwen3.5]: replaced {n} RMSNorm layers "
            "(Qwen3_5RMSNormGated inside DeltaNet intentionally skipped)"
        )

    if fuse_swiglu:
        n = _replace_mlps_qwen3_5(model)
        stats['swiglu'] = n
        logger.info(f"MetalLiger[Qwen3.5]: replaced {n} MLP layers with fused SwiGLU")

    if fuse_lora:
        n = patch_fused_lora(model)
        stats['fused_lora'] = n
        logger.info(f"MetalLiger[Qwen3.5]: fused LoRA marked {n} modules")
    else:
        logger.info(
            "MetalLiger[Qwen3.5]: fused LoRA DISABLED (default). Using standard PEFT backward "
            "for numerical stability. Set fuse_lora=True to re-enable (saves ~1.8GB but "
            "risks precision drift on long runs)."
        )

    # Phase 7: GatedDeltaNet Precision Hardening
    n_deltanet = patch_deltanet(model)
    stats['fused_deltanet'] = n_deltanet
    logger.info(f"MetalLiger[Qwen3.5]: hardened precision for {n_deltanet} DeltaNet modules")

    total = sum(v for v in stats.values() if isinstance(v, int))
    logger.info(
        f"MetalLiger[Qwen3.5]: {total} total layers patched. Details: {stats}"
    )

    return model
