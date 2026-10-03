# Copyright 2025 MetalLiger Contributors
"""Fused Linear + Cross-Entropy Loss — eliminates massive logit tensor.

THE BIGGEST WIN IN MetalLiger.

Standard PyTorch:
  logits = Linear(hidden, vocab_size)  → [batch, seq, 152064] = ~740MB!
  loss = F.cross_entropy(logits, labels)
  Peak memory: 740MB for the logit tensor

MetalLiger FusedLinearCrossEntropy:
  Computes loss in chunks of vocab_size, never materializing all logits at once.
  Peak memory: chunk_size × element_size = ~32KB per chunk

FIXES vs previous version:
  - Forward: replaced O(N_tokens) Python loop (one GPU dispatch / token) with
    vectorized index_select → single matmul for all correct-class logits.
  - Backward: now uses log_sum_exp saved from forward instead of recomputing it
    (saves 2 full passes over the weight matrix on every backward).
  - SwiGLU: removed unnecessary `up` tensor from saved_tensors.

References:
  - Liger Kernel FusedLinearCrossEntropy (Triton)
  - TRL mps_fused_loss.py (existing Python implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os


def _chunked_forward_pass(hidden, weight, bias, vocab_size, chunk_size, running_max, running_sum_exp):
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        b_chunk = bias[start:end] if bias is not None else None
        # Native precision GEMM on MPS (2x faster hardware BF16/FP16), upcasted to float32 for reduction
        chunk = F.linear(hidden, weight[start:end], b_chunk).float()  # [N, chunk]

        chunk_max = chunk.max(dim=-1).values  # [N]
        new_max = torch.maximum(running_max, chunk_max)

        # Rescale previous sum_exp to the new max, then add this chunk
        running_sum_exp = (
            running_sum_exp * torch.exp(running_max - new_max)
            + torch.exp(chunk - new_max.unsqueeze(-1)).sum(dim=-1)
        )
        running_max = new_max
    return running_max, running_sum_exp


def _chunked_backward_pass(hidden, weight, bias, labels, float_mask_2d, scale, log_sum_exp, vocab_size, chunk_size, grad_hidden, grad_weight):
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        w_chunk = weight[start:end]               # [chunk, hidden] in native precision
        b_chunk = bias[start:end] if bias is not None else None

        logit_chunk = F.linear(hidden, w_chunk, b_chunk).float()  # [N, chunk]

        # Softmax probabilities for this vocab chunk
        prob_chunk = (logit_chunk - log_sum_exp.unsqueeze(-1)).exp()  # [N, chunk]

        # Subtract 1 from the correct label position: grad = softmax - one_hot
        chunk_classes = torch.arange(start, end, device=labels.device, dtype=labels.dtype).unsqueeze(0)  # [1, chunk]
        is_correct_class = (labels.unsqueeze(1) == chunk_classes)  # [N, chunk] bool
        prob_chunk = torch.where(is_correct_class, prob_chunk - 1.0, prob_chunk)

        # Zero out padding/ignored positions using dense float multiplication
        prob_chunk = prob_chunk * float_mask_2d * scale

        # Accumulate gradients in native precision (BF16 hardware GEMM)
        prob_chunk_native = prob_chunk.to(hidden.dtype)
        grad_hidden.addmm_(prob_chunk_native, w_chunk)
        grad_weight[start:end] += prob_chunk_native.t() @ hidden
    return grad_hidden, grad_weight

class _FusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Chunked Linear + CrossEntropy that avoids materializing full logits.

    Strategy:
      1. Compute logits in chunks along the vocab dimension in native BF16/FP16
      2. For each chunk, compute partial softmax + CE contribution in FP32
      3. Use online softmax (log-sum-exp) to combine chunks numerically stably
      4. Save log_sum_exp for backward — do NOT recompute.
      5. Backward: recompute logit chunks and use softmax - one_hot for gradient.
    """

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor,
                labels: torch.Tensor, bias: torch.Tensor = None,
                ignore_index: int = -100, chunk_size: int = 8192):
        """
        Args:
            hidden: [batch * seq, hidden_dim] — last hidden state
            weight: [vocab_size, hidden_dim] — lm_head weight
            labels: [batch * seq] — target token IDs
            bias: Optional [vocab_size] — lm_head bias
            ignore_index: Label value to ignore in loss (default: -100)
            chunk_size: Vocab chunk size for memory efficiency
        """
        vocab_size = weight.shape[0]
        N = hidden.shape[0]

        # ── Single-pass online log-sum-exp in native BF16 matmul, FP32 accumulator ──
        running_max = torch.full((N,), float('-inf'), device=hidden.device, dtype=torch.float32)
        running_sum_exp = torch.zeros(N, device=hidden.device, dtype=torch.float32)

        running_max, running_sum_exp = _chunked_forward_pass(
            hidden, weight, bias, vocab_size, chunk_size, running_max, running_sum_exp
        )

        log_sum_exp = running_max + running_sum_exp.log()  # [N]

        # ── Gather correct-class logits — VECTORIZED index_select in native precision ──
        mask = labels != ignore_index
        
        # Dense mapping: avoids ANY boolean indexing which dynamically resizes tensors
        safe_labels = torch.where(mask, labels, 0)
        
        label_weight = weight[safe_labels]       # [N, hidden_dim]
        label_bias   = bias[safe_labels] if bias is not None else None

        # Elementwise dot product in float32 for loss precision
        correct_logits = (hidden.float() * label_weight.float()).sum(dim=-1) # [N]
        if label_bias is not None:
            correct_logits = correct_logits + label_bias.float()
            
        token_losses = log_sum_exp - correct_logits   # [N] CE per token
        
        # Zero out padding/ignored tokens
        token_losses = token_losses * mask.float()
        
        # Safe sync-free mean computation
        total_loss = token_losses.sum() / torch.clamp(mask.float().sum(), min=1.0)

        # Save log_sum_exp and mask for backward — eliminates full 2-pass recompute
        ctx.save_for_backward(hidden, weight, labels, log_sum_exp, mask)
        ctx.bias = bias
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size

        return total_loss.squeeze()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, labels, log_sum_exp, mask = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        bias = ctx.bias
        vocab_size = weight.shape[0]

        # Sync-free scale calculation
        num_valid = torch.clamp(mask.sum().float(), min=1.0)

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)
        scale = (grad_output / num_valid).float()

        # Pre-compute the float mask for dense multiplication (no boolean indexing)
        float_mask = mask.float()  # [N]
        float_mask_2d = float_mask.unsqueeze(-1)  # [N, 1] — broadcasts to [N, chunk]

        grad_hidden, grad_weight = _chunked_backward_pass(
            hidden, weight, bias, labels, float_mask_2d, scale, log_sum_exp,
            vocab_size, chunk_size, grad_hidden, grad_weight
        )

        return (grad_hidden, grad_weight, None, None, None, None)


class MetalLigerFusedLinearCrossEntropy(nn.Module):
    """Fused lm_head + cross-entropy loss.

    Replaces:
        logits = lm_head(hidden_states)  # [batch, seq, 152064] — 740MB!
        loss = F.cross_entropy(logits, labels)

    With chunked computation that never materializes the full logit tensor.

    Args:
        lm_head: The language model head linear layer
        ignore_index: Label ID to ignore (default: -100)
        chunk_size: Vocab chunk size (default: 8192)
    """

    def __init__(self, lm_head: nn.Linear, ignore_index: int = -100,
                 chunk_size: int = 8192):
        super().__init__()
        self.weight = lm_head.weight
        self.bias = lm_head.bias
        self.ignore_index = ignore_index
        self.chunk_size = chunk_size

    def forward(self, hidden_states: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss without materializing full logits.

        Args:
            hidden_states: [batch, seq, hidden] — last hidden state
            labels: [batch, seq] — target token IDs

        Returns:
            Scalar loss tensor
        """
        batch_seq = (hidden_states.shape[0] * hidden_states.shape[1]
                     if hidden_states.ndim == 3 else hidden_states.shape[0])
        hidden_flat = hidden_states.reshape(batch_seq, -1)
        labels_flat = labels.reshape(-1)

        return _FusedLinearCrossEntropyFunction.apply(
            hidden_flat, self.weight, labels_flat,
            self.bias, self.ignore_index, self.chunk_size
        )
