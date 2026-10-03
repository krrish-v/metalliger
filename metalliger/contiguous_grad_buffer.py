# Copyright 2025 Metalliger Contributors
"""Contiguous Gradient Buffer for MPS — eliminates gradient memory fragmentation.

Ported from DeepSpeed's BF16 Optimizer (bf16_optimizer.py) approach of using
a single flat buffer with .narrow() views for all parameter gradients.

On Apple Silicon, hundreds of small scattered gradient allocations cause:
  1. Page-table pressure in the unified memory system
  2. MPS allocator fragmentation (holes between allocations)
  3. Increased TLB misses during gradient accumulation

This module pre-allocates ONE contiguous buffer and maps each parameter's
gradient to a slice of it. The optimizer sees the same .grad tensors — only
the memory layout changes.

ACCURACY IMPACT: Zero — identical gradient values, just contiguous in memory.
CORRUPTION RISK: None — only touches .grad, never touches .data (weights).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ContiguousGradBuffer:
    """Pre-allocated flat buffer for all trainable parameter gradients.

    Usage:
        buffer = ContiguousGradBuffer(model, dtype=torch.float32)
        # After this, all param.grad tensors are views into one flat buffer.
        # Call buffer.zero_() to zero all gradients in a single op.
        # Call buffer.release() to detach and free the buffer.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("mps")
        self._hooks = []
        self._param_views = {}
        self._flat_buffers = {}

        # Collect all trainable parameters
        trainable_params = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]

        if not trainable_params:
            logger.warning("ContiguousGradBuffer: no trainable parameters found")
            return

        # PyTorch requires param.grad to have the exact same dtype as the parameter.
        # Since models (especially with PEFT/LoRA) can have mixed dtypes (e.g. bfloat16 and float32),
        # we group parameters by their dtype and allocate ONE flat buffer PER dtype.
        params_by_dtype = {}
        for name, p in trainable_params:
            if p.dtype not in params_by_dtype:
                params_by_dtype[p.dtype] = []
            params_by_dtype[p.dtype].append((name, p))

        for dtype, params in params_by_dtype.items():
            total_numel = sum(p.numel() for _, p in params)
            flat_buffer = torch.zeros(total_numel, dtype=dtype, device=self.device)
            self._flat_buffers[dtype] = flat_buffer
            
            offset = 0
            for name, param in params:
                numel = param.numel()
                grad_view = flat_buffer.narrow(0, offset, numel).view(param.shape)
                
                self._param_views[name] = (param, grad_view, offset, numel)

                def _make_hook(p, view, target_dtype):
                    def _grad_hook(grad):
                        # Re-attach view if detached by optimizer.zero_grad(set_to_none=True)
                        if p.grad is None or p.grad.data_ptr() != view.data_ptr():
                            view.zero_()       # Clear stale gradients from previous optimizer step
                            p.grad = view      # Re-attach to our contiguous buffer
                        # Return cast gradient — PyTorch accumulates it into p.grad (= view) exactly ONCE.
                        # DO NOT also call view.add_() here — that would double-count the gradient!
                        return grad.to(target_dtype)
                    return _grad_hook

                hook = param.register_hook(_make_hook(param, grad_view, dtype))
                self._hooks.append(hook)
                
                # Pre-set the .grad to point to our buffer
                param.grad = grad_view
                offset += numel

            buffer_mb = (total_numel * flat_buffer.element_size()) / (1024 * 1024)
            logger.info(
                f"ContiguousGradBuffer: allocated {buffer_mb:.1f}MB contiguous buffer "
                f"for {len(params)} parameters with dtype {dtype} ({total_numel:,} elements)"
            )

    @property
    def flat_buffers(self) -> dict:
        """The underlying dict of flat buffer tensors."""
        return self._flat_buffers

    def zero_(self):
        """Zero all gradients in a single operation per dtype."""
        for buffer in self._flat_buffers.values():
            buffer.zero_()

    def zero_foreach(self):
        """Zero all gradients. Alternative to zero_()."""
        for buffer in self._flat_buffers.values():
            buffer.zero_()

    def grad_norm(self, norm_type: float = 2.0) -> torch.Tensor:
        """Compute gradient norm over the entire contiguous buffer."""
        if not self._flat_buffers:
            return torch.tensor(0.0, device=self.device)
        
        total_norm = torch.tensor(0.0, device=self.device)
        for buffer in self._flat_buffers.values():
            norm = buffer.float().norm(norm_type)
            total_norm += norm ** norm_type
            
        return total_norm ** (1.0 / norm_type)

    def release(self):
        """Remove all hooks and release the buffer."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._param_views.clear()
        self._flat_buffers.clear()
        logger.info("ContiguousGradBuffer: released")

    def __del__(self):
        self.release()