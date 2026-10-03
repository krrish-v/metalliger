# Copyright 2025 MetalLiger Contributors
"""Native Metal Generator for Apple Silicon.

Optimized token-by-token generation for unified memory.
Eliminates overhead by using a static KV-Cache and minimizing
Python->C++ crossings.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union

try:
    from transformers.cache_utils import Cache
except ImportError:
    class Cache: pass

class MetalLigerStaticCache(Cache):
    """Static KV-cache for MPS that fits the Transformers Cache API.
    Uses a list of per-layer 4D tensors to avoid 5D slicing overhead on MPS.
    """
    
    def __init__(self, model, batch_size, max_seq_len):
        try:
            super().__init__()
        except (ValueError, TypeError):
            pass
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.config = model.config
        text_config = getattr(self.config, "text_config", self.config)
        
        self.num_layers = getattr(text_config, "num_hidden_layers", getattr(self.config, "num_hidden_layers", 28))
        self.num_heads = getattr(text_config, "num_attention_heads", getattr(self.config, "num_attention_heads", 16))
        self.num_kv_heads = getattr(text_config, "num_key_value_heads", getattr(self.config, "num_key_value_heads", self.num_heads))
        hidden_size = getattr(text_config, "hidden_size", getattr(self.config, "hidden_size", 2048))
        self.head_dim = hidden_size // self.num_heads
        
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        
        # Per-layer 4D buffers: [batch, num_kv_heads, max_seq_len, head_dim]
        self.key_cache = [
            torch.zeros((batch_size, self.num_kv_heads, max_seq_len, self.head_dim),
                        device=self.device, dtype=self.dtype)
            for _ in range(self.num_layers)
        ]
        self.value_cache = [
            torch.zeros((batch_size, self.num_kv_heads, max_seq_len, self.head_dim),
                        device=self.device, dtype=self.dtype)
            for _ in range(self.num_layers)
        ]
        self.seen_tokens = 0

    @property
    def batch_size(self):
        return getattr(self, "_batch_size", 1)

    def reset(self):
        """Zero the cache for reuse without re-allocating memory."""
        self.seen_tokens = 0
        for k, v in zip(self.key_cache, self.value_cache):
            k.zero_()
            v.zero_()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Update the cache and return the full history."""
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        start = self.seen_tokens
        end = start + seq_len
        
        k_buf = self.key_cache[layer_idx]
        v_buf = self.value_cache[layer_idx]

        k_buf[:, :, start:end, :] = key_states
        v_buf[:, :, start:end, :] = value_states
        
        # Return slice up to current length
        return k_buf[:, :, :end, :], v_buf[:, :, :end, :]

    def get_seq_length(self, layer_idx=0):
        return self.seen_tokens

    def get_max_length(self):
        return self._max_seq_len

class MetalLigerGenerator:
    """High-speed token generator optimized for MPS."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.tokenizer = None # Should be set if used standalone

    @torch.inference_mode()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Optimized generation loop for MPS without dynamic reallocation."""
        batch_size, prompt_len = input_ids.shape
        max_len = prompt_len + max_new_tokens
        
        # 1. Initialize Static Cache
        cache = MetalLigerStaticCache(self.model, batch_size, max_len)
        
        # Multimodal M-RoPE position tracking (e.g. Qwen3VL)
        position_ids = None
        if hasattr(self.model, "get_rope_index"):
            try:
                position_ids, _ = self.model.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=kwargs.get("image_grid_thw", None),
                    attention_mask=kwargs.get("attention_mask", None),
                )
                kwargs["position_ids"] = position_ids
            except Exception:
                position_ids = kwargs.get("position_ids", None)
        else:
            position_ids = kwargs.get("position_ids", None)

        cur_position_ids = position_ids[..., -1:] + 1 if position_ids is not None else None

        # Filter out multimodal kwargs for single-token decode steps
        decode_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "position_ids")
        }

        # Pre-allocate output buffer to completely eliminate torch.cat reallocations
        out_ids = torch.empty((batch_size, max_len), dtype=torch.long, device=input_ids.device)
        out_ids[:, :prompt_len] = input_ids
        cur_len = prompt_len

        # 2. Burst Prompt (Prefill)
        outputs = self.model(
            input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            **kwargs
        )
        cache.seen_tokens += prompt_len
        
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        out_ids[:, cur_len] = next_token.squeeze(-1)
        cur_len += 1
        
        # Fast scalar check for batch_size == 1 to avoid GPU-CPU sync stalls
        is_single_batch = (batch_size == 1)

        # 3. Token-by-token Decode
        for i in range(max_new_tokens - 1):
            if eos_token_id is not None:
                if is_single_batch:
                    if int(next_token[0, 0]) == eos_token_id:
                        break
                elif (next_token == eos_token_id).all():
                    break
                
            step_kwargs = dict(decode_kwargs)
            if cur_position_ids is not None:
                step_kwargs["position_ids"] = cur_position_ids

            outputs = self.model(
                next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                **step_kwargs
            )
            cache.seen_tokens += 1
            if cur_position_ids is not None:
                cur_position_ids = cur_position_ids + 1
            
            logits = outputs.logits[:, -1, :]
            
            if temperature == 0 or (temperature == 1.0 and top_p == 1.0):
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            out_ids[:, cur_len] = next_token.squeeze(-1)
            cur_len += 1
            
        return out_ids[:, :cur_len]

def patch_generator(model: nn.Module, force: bool = False):
    """Replace model.generate with MetalLigerGenerator.generate.
    
    WARNING: The custom generator does NOT support:
      - do_sample=True (no temperature/top_p/top_k sampling)
      - GenerationConfig 
      - attention_mask / pad_token_id handling
      - Any standard HF generate kwargs
    
    This means it is INCOMPATIBLE with GRPO/RL training which relies on 
    Transformers' native generate() for sampling. Only use for standalone
    greedy inference.
    
    Args:
        model: Model to patch
        force: If True, patch even if model.training is True (unsafe for GRPO)
    """
    if model.training and not force:
        import logging
        logging.getLogger(__name__).info(
            "MetalLiger: Skipping generator patch — model is in training mode. "
            "The custom generator is incompatible with GRPO/RL sampling. "
            "Use force=True to override."
        )
        return model
    
    generator = MetalLigerGenerator(model)
    # We store the original generate just in case
    if not hasattr(model, "_orig_generate"):
        model._orig_generate = model.generate
    
    # Monkey-patch
    model.generate = generator.generate
    return model
