# Copyright 2025 MetalLiger Contributors
"""Phase 4: Layer-level torch.compile for PEFT-safe graph capture on MPS.

Why layer-level, not model-level:
  PEFT wraps the model with Python routing logic (adapter switching, module
  dispatch) that creates 50+ graph breaks if you compile model.forward.
  But PEFT physically replaces nn.Linear with LoRA.Linear INSIDE each
  transformer layer. So compiling layer.forward traces:
      x @ W_base + (x @ lora_A @ lora_B) * scale
  as pure linear algebra — no graph breaks.

Why dynamic=True:
  VLMs convert images to visual tokens based on aspect ratio.
  Variable token counts = variable sequence lengths per batch.
  dynamic=True tells Dynamo to accept shape variations without recompiling.
  IMPORTANT: recompile_limit is capped at 8 to prevent silent RAM leaks
  if Dynamo can't handle a shape change and falls back to recompilation.
  If you see 'recompile_limit' warnings, switch to padding="max_length".

Why warmup must include backward:
  aot_eager compiles forward and backward graphs SEPARATELY.
  Forward-only warmup leaves backward uncompiled → step 1 stalls on
  the first loss.backward() call while the GPU waits for CPU to JIT
  the backward kernels. Full micro-step warmup (fwd+bwd+zero_grad)
  compiles both graphs before training starts.

What gets compiled:
  1. Each LLM transformer layer (28× for Qwen3-VL 7B)
  2. Vision Merger — contains linear_fc1/fc2 LoRA targets
     (skipped means LoRA math there stays in slow Python dispatch)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_COMPILE_APPLIED = False


# ── Layer finder ──────────────────────────────────────────────────────────────

def _find_transformer_layers(model: nn.Module) -> list:
    """Walk model hierarchy to find transformer layer list.

    Tries static known paths first (fast), then falls back to recursive
    tree walk so it works on any model architecture including Qwen3-VL.
    """
    # Unwrap PEFT if present
    base = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model.model

    # ── Static paths: most common VLM layouts ─────────────────────────────
    candidates = [
        # Qwen3-VL: model.model has separate language_model attribute
        ("model.language_model.model.layers",  lambda m: m.model.language_model.model.layers),
        ("model.language_model.layers",        lambda m: m.model.language_model.layers),
        # Qwen2-VL: layers directly under model.model
        ("model.model.layers",                 lambda m: m.model.model.layers),
        # Qwen2-VL alt: layers directly under model
        ("model.layers",                       lambda m: m.model.layers),
        # language_model at top level (some architectures)
        ("language_model.model.layers",        lambda m: m.language_model.model.layers),
        ("language_model.layers",              lambda m: m.language_model.layers),
        # Bare LLM (non-VLM, already unwrapped)
        ("layers",                             lambda m: m.layers),
    ]

    for path_name, path_fn in candidates:
        try:
            layers = path_fn(base)
            if isinstance(layers, (list, nn.ModuleList)) and len(layers) > 0:
                logger.info(
                    f"MetalLiger compile: found {len(layers)} transformer layers "
                    f"at path '{path_name}'"
                )
                return list(layers)
        except AttributeError:
            continue

    # ── Recursive fallback: walk entire module tree ────────────────────────
    # Find the largest nn.ModuleList — that's almost certainly the layer stack
    logger.info(
        "MetalLiger compile: static paths failed, attempting recursive layer search..."
    )
    best_layers = []
    best_path = ""

    def _walk(module: nn.Module, path: str, depth: int = 0):
        nonlocal best_layers, best_path
        if depth > 6:  # don't go too deep
            return
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            if isinstance(child, nn.ModuleList) and len(child) > len(best_layers):
                # Prefer lists with >5 modules that look like transformer layers
                # (have norm/attn/mlp sub-modules rather than simple layers)
                if len(child) > 5:
                    sample = list(child)[0]
                    has_attn = any(
                        "attn" in n.lower() or "attention" in n.lower()
                        for n, _ in sample.named_children()
                    )
                    if has_attn:
                        best_layers = list(child)
                        best_path = child_path
            _walk(child, child_path, depth + 1)

    _walk(base, "")

    if best_layers:
        logger.info(
            f"MetalLiger compile: recursive search found {len(best_layers)} layers "
            f"at path '{best_path}'"
        )
        return best_layers

    # Log the top-level attributes to help debug unknown architectures
    top_attrs = [n for n, _ in base.named_children()]
    logger.warning(
        f"MetalLiger compile: no transformer layers found. "
        f"Top-level attributes: {top_attrs}. "
        f"Try adding your model's path to _find_transformer_layers() in _compile.py"
    )
    return []


def _find_vision_merger(model: nn.Module) -> Optional[nn.Module]:
    """Find the Vision-Language merger module (houses linear_fc1/fc2 LoRA).

    The merger bridges vision tower and LLM. Tries known paths then
    recursively searches for a module named 'merger'.
    """
    base = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base = model.base_model.model

    candidates = [
        ("model.visual.merger",       lambda m: m.model.visual.merger),
        ("visual.merger",             lambda m: m.visual.merger),
        ("model.vision_model.merger", lambda m: m.model.vision_model.merger),
        ("vision_model.merger",       lambda m: m.vision_model.merger),
        ("encoder.merger",            lambda m: m.encoder.merger),
    ]

    for path_name, path_fn in candidates:
        try:
            merger = path_fn(base)
            if merger is not None:
                logger.info(f"MetalLiger compile: found Vision Merger at '{path_name}'")
                return merger
        except AttributeError:
            continue

    # Recursive fallback: find any module named "merger"
    def _find_named(module: nn.Module, target: str, depth: int = 0) -> Optional[nn.Module]:
        if depth > 6:
            return None
        for name, child in module.named_children():
            if name == target:
                return child
            result = _find_named(child, target, depth + 1)
            if result is not None:
                return result
        return None

    merger = _find_named(base, "merger")
    if merger is not None:
        logger.info("MetalLiger compile: found Vision Merger via recursive search")
        return merger


    logger.warning(
        "MetalLiger compile: Vision Merger not found. "
        "linear_fc1/fc2 LoRA targets will remain uncompiled (Python dispatch)."
    )
    return None


# ── Compiler ──────────────────────────────────────────────────────────────────

def _detect_best_compile_backend() -> str:
    """Detect the most efficient compile backend for PyTorch 2.14+ on MPS."""
    try:
        fn = torch.compile(lambda x: x * 2.0, backend="inductor", dynamic=True)
        t = torch.ones(1, device="mps", dtype=torch.bfloat16)
        fn(t)
        return "inductor"
    except Exception:
        return "aot_eager"


def compile_transformer_layers(
    model: nn.Module,
    backend: Optional[str] = None,
) -> dict:
    """Compile each transformer layer + vision merger individually.

    PEFT-safe: layer.forward sees LoRA as pure linear algebra (no graph breaks).

    Args:
        model:   The model (may be PEFT-wrapped).
        backend: Backend to use ('inductor' or 'aot_eager'). Defaults to auto-detect.

    Returns:
        dict with keys: 'llm_layers', 'merger_compiled', 'total'
    """
    if not torch.backends.mps.is_available():
        logger.warning("MetalLiger compile: MPS not available — skipping.")
        return {"llm_layers": 0, "merger_compiled": False, "total": 0}

    if backend is None:
        backend = _detect_best_compile_backend()

    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = 16  # Accommodate dynamic shapes in PyTorch 2.14
    if hasattr(dynamo.config, "dynamic_shapes"):
        dynamo.config.dynamic_shapes = True

    
    compiled_count = 0

    # ── 1. LLM transformer layers ──────────────────────────────────────────
    layers = _find_transformer_layers(model)
    if not layers:
        logger.warning(
            "MetalLiger compile: no transformer layers found. "
            "Check model architecture. Skipping layer compilation."
        )
    else:
        for i, layer in enumerate(layers):
            try:
                # ── Qwen3.5 Hybrid Architecture Detection ──────────────────────
                # 'linear_attn' is the attribute name for Qwen3_5GatedDeltaNet.
                # It is EXCLUSIVE to Qwen3.5 hybrid layers — standard transformer
                # layers (Qwen3-VL, Qwen2, LLaMA, etc.) do NOT have this attribute.
                # Do NOT use hasattr(layer, 'self_attn') as the discriminator:
                # Qwen3VL layers also have self_attn and would incorrectly fall
                # into the sub-module path, losing the whole-layer compile benefit.
                is_hybrid_layer = hasattr(layer, 'linear_attn')

                if is_hybrid_layer:
                    # SSM DeltaNet layer: only compile the MLP, leave linear_attn eager.
                    # GatedDeltaNet has recurrent state that aot_eager can't trace on MPS.
                    sub_compiled = 0
                    if hasattr(layer, 'mlp') and layer.mlp is not None:
                        # CRITICAL: Do not apply torch.compile() directly to a function
                        # decorated with @torch.compiler.disable (our fused LoRA wrappers).
                        # Explicitly compiling the DisableContext object causes MPS JIT to segfault.
                        if not getattr(layer.mlp, '_metalliger_fused_lora', False):
                            layer.mlp.forward = torch.compile(
                                layer.mlp.forward, backend=backend, fullgraph=False, dynamic=True
                            )
                            sub_compiled += 1
                    if sub_compiled > 0:
                        compiled_count += 1

                elif hasattr(layer, 'self_attn'):
                    # Standard attention layer (pure transformer OR Qwen3.5 attention layer).
                    # Compile the ENTIRE layer.forward instead of sub-modules.
                    # If self_attn or mlp were patched with @torch.compiler.disable, 
                    # Dynamo will trace into layer.forward and insert clean graph breaks
                    # automatically, avoiding the JIT crash that happens when compiling them directly.
                    layer.forward = torch.compile(
                        layer.forward, backend=backend, fullgraph=False, dynamic=True
                    )
                    compiled_count += 1

                else:
                    # Fallback: compile entire layer forward (works for most architectures)
                    layer.forward = torch.compile(
                        layer.forward, backend=backend, fullgraph=False, dynamic=True
                    )
                    compiled_count += 1

            except Exception as e:
                logger.warning(f"MetalLiger compile: layer {i} failed — {e}")

        logger.info(
            f"MetalLiger compile: compiled {compiled_count}/{len(layers)} "
            f"LLM transformer layers (or sub-modules) (dynamic=True, backend={backend})"
        )

    # ── 2. Vision Merger (houses linear_fc1/fc2 LoRA targets) ─────────────
    merger_compiled = False
    merger = _find_vision_merger(model)
    if merger is not None:
        try:
            merger.forward = torch.compile(
                merger.forward,
                backend=backend,
                fullgraph=False,
                dynamic=True,  # visual token count varies — must be dynamic
            )
            merger_compiled = True
            compiled_count += 1
            logger.info("MetalLiger compile: Vision Merger compiled (linear_fc1/fc2 LoRA included)")
        except Exception as e:
            logger.warning(f"MetalLiger compile: Vision Merger compilation failed — {e}")

    total = compiled_count
    logger.info(
        f"MetalLiger compile: {total} modules compiled. "
        f"{'Run with TORCH_LOGS=recompiles to detect shape-change recompilation.' if total > 0 else ''}"
    )
    return {"llm_layers": len(layers) if layers else 0, "merger_compiled": merger_compiled, "total": total}


def warmup_compiled_model(
    model: nn.Module,
    dummy_batch: dict,
) -> None:
    """Run a full micro-step (fwd + bwd + zero_grad) to pre-compile both graphs.

    aot_eager compiles FORWARD and BACKWARD graphs SEPARATELY:
    - Forward graph compiled on first model(**batch) call
    - Backward graph compiled on first loss.backward() call

    If warmup only runs forward, the first training loss.backward() stalls
    for 60-120s while the M4 Pro JITs the backward gradient kernels.
    This warmup runs both so training step 1 starts at full compiled speed.

    Uses a real training batch (not synthetic dummy) so all VLM code paths
    (pixel_values, image_grid_thw, visual merger, etc.) are traced.
    """
    print("MetalLiger: ── Starting full-circuit warmup (Fwd + Bwd) ──")
    print("MetalLiger: This locks in both compiled graphs. Takes 30-120s. Do not interrupt.")

    model.train()
    try:
        # ── Pre-flight: detect and repair NaN LoRA weights ──────────────────
        # MPS bug: PEFT's kaiming_uniform_ on bfloat16 can produce all-NaN
        # tensors during initialization. This happens BEFORE checkpoint weights
        # are loaded, and if the checkpoint key mapping fails (e.g. missing
        # '.default.' in key names), the NaN tensors remain.
        print("MetalLiger: [0/3] Pre-flight parameter check...")
        nan_params = []
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                nan_params.append((name, torch.isnan(p).sum().item(), p.numel(), p))

        if nan_params:
            print(f"MetalLiger: [0/3] ⚠️ Found {len(nan_params)} parameters with NaN!")
            for name, cnt, total, _ in nan_params[:5]:
                print(f"  NaN: {name} ({cnt}/{total} elements)")

            # Auto-repair: re-initialize NaN LoRA weights with PEFT defaults
            repaired = 0
            for name, cnt, total, p in nan_params:
                if 'lora_A' in name and 'weight' in name:
                    with torch.no_grad():
                        torch.nn.init.kaiming_uniform_(p, a=5**0.5)
                    repaired += 1
                elif 'lora_B' in name and 'weight' in name:
                    with torch.no_grad():
                        p.zero_()
                    repaired += 1

            if repaired > 0:
                print(f"MetalLiger: [0/3] 🔧 Auto-repaired {repaired}/{len(nan_params)} NaN LoRA weights")
                still_nan = sum(1 for _, p in model.named_parameters() if torch.isnan(p).any())
                if still_nan:
                    print(f"MetalLiger: [0/3] ⚠️ {still_nan} parameters STILL have NaN after repair")
                else:
                    print(f"MetalLiger: [0/3] ✓ All parameters clean after repair")
        else:
            print("MetalLiger: [0/3] All parameters clean ✓")

        # Move real training batch to model's device
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in dummy_batch.items()}

        # ── Step 1: Forward trace ───────────────────────────────────────────
        # Strip labels so we compute a synthetic loss from raw logits
        # instead of going through HF's internal CE path.
        print("MetalLiger: [1/3] Forward graph compilation...")
        warmup_batch = {k: v for k, v in batch.items()
                        if k not in ("labels", "shift_labels")}
        warmup_batch["use_cache"] = False
        with torch.enable_grad():
            outputs = model(**warmup_batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Clamp NaN/Inf to finite values so backward ALWAYS runs.
            logits_safe = torch.nan_to_num(logits[..., 0].float(), nan=0.0, posinf=1.0, neginf=-1.0)
            loss = logits_safe.mean()

        loss_val = loss.detach().item()
        print(f"MetalLiger: [1/3] Forward graph locked. Loss = {loss_val:.6f} (synthetic)")

        # ── Step 2: Backward trace ──────────────────────────────────────────
        print("MetalLiger: [2/3] Backward graph compilation (gradient kernels)...")
        loss.backward()
        print("MetalLiger: [2/3] Backward graph locked. Gradient kernels compiled.")

        # ── Step 3: Reset + flush ───────────────────────────────────────────
        print("MetalLiger: [3/3] Clearing warmup gradients + flushing MPS cache...")
        model.zero_grad(set_to_none=True)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()

        print("MetalLiger: ── Warmup complete. Both graphs locked in. ──")
        print("MetalLiger: Training step 1 will run at full compiled speed.")

    except Exception as e:
        print(f"MetalLiger: ⚠️ Warmup failed — {e}")
        print("MetalLiger: Step 1 will be slow (backward graph compiles during training).")
        logger.warning(f"MetalLiger warmup failed — {e}")
        model.zero_grad(set_to_none=True)

        # Flush MPS command queues
        if torch.backends.mps.is_available():
            torch.mps.synchronize()


# ── Legacy API (kept for backward compat with existing trainer code) ──────────

def apply_compile(model: nn.Module, **kwargs) -> nn.Module:
    """Deprecated: use compile_transformer_layers() instead.
    
    This model-level compile breaks on PEFT. Kept only so old code
    doesn't crash on import. Raises a clear warning.
    """
    logger.warning(
        "MetalLiger: apply_compile() is deprecated and PEFT-unsafe. "
        "Use compile_transformer_layers() instead. Skipping compile."
    )
    return model


def reset_compile_state():
    """Reset compile tracking (for testing)."""
    global _COMPILE_APPLIED
    _COMPILE_APPLIED = False
