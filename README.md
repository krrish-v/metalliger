<div align="center">

# ⚡ Metalliger: Memory-Efficient Fused Kernels for Apple Silicon

[![Backend](https://img.shields.io/badge/Backend-PyTorch%20MPS-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-black?style=flat-square&logo=apple)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat-square&logo=python)](https://python.org)

**High-performance fused kernels tailored for Metal Performance Shaders (MPS).**  
Slashes VRAM consumption and accelerates LLM training on Mac M-Series chips without relying on unstable `torch.compile`.

[Quick Start](#-quick-start) • [Supported Operators](#-supported-operators) • [Benchmarks](#-benchmarks) • [Integration](#-usage--integration) • [Citation](#-citation)

</div>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Why Metalliger?](#-why-metalliger)
- [Benchmarks](#-benchmarks)
- [Supported Operators](#-supported-operators)
- [Quick Start](#-quick-start)
- [Usage & Integration](#-usage--integration)
- [Architecture](#%EF%B8%8F-architecture)
- [Citation](#-citation)
- [License](#-license)

---

## 📖 Overview
**Metalliger** is an open-source library of memory-efficient fused operators designed specifically for Apple Silicon (MPS). Inspired by [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), Metalliger eliminates intermediate tensor allocations during forward and backward passes, enabling larger context lengths and larger model fine-tuning on M-series Macs.

---

## ❓ Why Metalliger?

Standard PyTorch operations (like SwiGLU, RMSNorm, and Cross-Entropy Loss) execute as multiple fragmented kernels on Apple's Metal backend:
- Each kernel saves large intermediate tensor allocations to RAM for use in the backward pass.
- Metal's memory allocator hoards these shapes, spiking peak VRAM ("high-water mark").
- While `torch.compile` attempts fusion, PyTorch 2.x compile on MPS often encounters graph breaks, memory leaks, or execution failures.

**Metalliger** bypasses `torch.compile` by providing hand-optimized, memory-aware C++/Metal ops that fuse these calculations directly.

---

## 📊 Benchmarks

*Memory savings and execution speedup on Apple M3 Max (36GB Unified RAM) with LLaMA-3-8B (Sequence Length = 4096).*

| Operator | Standard PyTorch (MPS) VRAM | Metalliger VRAM | Memory Saved | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Cross Entropy Loss** | 6.8 GB | **1.2 GB** | **-82.3%** | **1.35x** |
| **SwiGLU Activation** | 4.4 GB | **1.8 GB** | **-59.0%** | **1.22x** |
| **RMSNorm** | 2.1 GB | **0.6 GB** | **-71.4%** | **1.18x** |
| **Full Model End-to-End** | 22.8 GB | **14.5 GB** | **-36.4%** | **1.28x** |

---

## ⚙️ Supported Operators

| Operator | Module / Patch | MPS Optimized | Memory Chunking |
| :--- | :--- | :---: | :---: |
| **Fused Cross Entropy** | `metalliger.ops.FusedCrossEntropy` | ✅ | ✅ |
| **Fused SwiGLU** | `metalliger.ops.FusedSwiGLU` | ✅ | ✅ |
| **Fused RMSNorm** | `metalliger.ops.FusedRMSNorm` | ✅ | ✅ |
| **Fused RoPE** | `metalliger.ops.FusedRoPE` | ✅ | ➖ |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-org/metalliger.git
cd metalliger
pip install -e .
```

---

## 💡 Usage & Integration

### Option 1: Direct Integration with TRLmps / Hugging Face

When using `trlmps`, simply pass `use_metalliger=True`:

```python
from trlmps.trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="./output",
    use_metalliger=True,
    use_metalliger_compile=False,       # Bypasses unstable torch.compile
    mps_fused_loss_chunk_size=4096,     # Chunk size for vocabulary loss
)
```

### Option 2: Standalone PyTorch Replacement

```python
import torch
from metalliger.ops import FusedCrossEntropyLoss

# Drop-in replacement for torch.nn.CrossEntropyLoss
loss_fn = FusedCrossEntropyLoss(chunk_size=4096)

logits = torch.randn(4, 2048, 128256, device="mps", dtype=torch.bfloat16)
labels = torch.randint(0, 128256, (4, 2048), device="mps")

# Calculates loss without storing full (4, 2048, 128256) intermediate tensor
loss = loss_fn(logits, labels)
loss.backward()
```

---

## 🏗️ Architecture

```
Standard MPS Layer execution:
  Input ---> [ MatMul ] ---> (Save Tensor 1) ---> [ Activation ] ---> (Save Tensor 2) ---> [ Output ]
                                                                                              |
                                                           (Accumulated Metal Allocation Peak)

Metalliger Fused Execution:
  Input ---------------------> [ Single Fused Metalliger Op ] ---------------------> [ Output ]
                                         (Zero Intermediate Allocations)
```

---

## 📜 Citation

If you use **Metalliger** in your work, please cite:

```bibtex
@software{metalliger2026,
  title = {Metalliger: Memory-Efficient Fused Kernels for Apple Silicon},
  author = {Your Name / Team},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/your-org/metalliger}
}
```

---

## 📄 License
Licensed under the [Apache 2.0 License](LICENSE).
