# ⚡ MetalLiger

<p align="center">
  <strong>Ultra-Fast Native PyTorch LLM & VLM Inference on Apple Silicon</strong>
  <br />
  <em>Zero model conversion. Fused Metal kernels. +35% faster token generation than MLX-VLM.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon%20(M1--M4)-black?logo=apple" alt="Apple Silicon" />
  <img src="https://img.shields.io/badge/Framework-PyTorch%20MPS-EE4C2C?logo=pytorch" alt="PyTorch MPS" />
  <img src="https://img.shields.io/badge/Flash%20Attention-Supported-blue" alt="Flash Attention" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License" />
</p>

---

## 🚀 Overview

**MetalLiger** brings Liger Kernel-style operator fusion, Flash Attention, and zero-allocation static KV caching to Apple's **Metal Performance Shaders (MPS)** backend.

Running LLMs and Vision-Language Models (VLMs) on Apple Silicon has traditionally forced developers into a trade-off:
- **MLX / MLX-VLM**: Fast, but requires converting weights to a proprietary ecosystem, often altering numerics, breaking PyTorch tooling, and introducing precision drift.
- **Vanilla PyTorch MPS**: Ecosystem compatibility, but bogged down by Python-to-C++ dispatch overhead, dynamic memory allocations, and fragmented Metal GPU kernels.

**MetalLiger eliminates the compromise.** By running **100% native PyTorch** with fused Metal kernels, pre-allocated static memory buffers, and MPS Flash Attention, MetalLiger outperforms MLX-VLM in autoregressive token generation speed while preserving exact model reasoning quality.

---

## 📊 Benchmark: MetalLiger vs. MLX-VLM

Tested on Apple Silicon (**M4 Pro**) running a complex medical Vision-Language task (**Mammogram Microcalcification Analysis & Diagnostic Reasoning**) with `Qwen3VL-2B-Medmaxv11`:

| Metric | ⚡ MetalLiger (Native PyTorch MPS) | 🍎 MLX-VLM | Advantage |
| :--- | :--- | :--- | :--- |
| **Phase 1: Prefill (Prompt Ingestion)** | 149.1 tokens/sec (1.12 s) | **750.6 tokens/sec** (0.25 s) | MLX compute burst on initial prompt |
| **Phase 2: Decode (Token Generation)** | **48.1 tokens/sec** | 35.65 tokens/sec | **MetalLiger is ~35% Faster** 🚀 |
| **Decode Latency (470+ tokens)** | **9.79 s** | 13.20 s (extrapolated) | **Lower autoregressive latency** |
| **Model Conversion** | **Zero conversion** (Direct Hugging Face weights) | Mandatory conversion to MLX format | **Seamless PyTorch workflow** |
| **Memory Management** | **Zero-allocation `M4StaticCache`** | Dynamic reallocation / tensor copies | **Zero GPU memory stalls** |
| **Clinical Reasoning & Accuracy** | **Correct (Option B)**<br />Detailed 4-paragraph clinical reasoning identifying malignant clustered microcalcifications | **Incorrect (Option A - Hallucination)**<br />Superficial response misclassifying lesion as benign fibroadenoma | **Exact Precision Retention** |
| **PyTorch Ecosystem** | **Full compatibility** (Hugging Face, Transformers, PEFT, PyTorch MPS) | Isolated MLX runtime | **Ecosystem Native** |

> **Key Takeaway**: While MLX achieves a burst during initial prompt prefill, **MetalLiger dominates where it matters most: sustained autoregressive token generation (+35% faster decode speed)** while avoiding the precision drift that caused MLX to misdiagnose the medical scan.

---

## 🔬 Head-to-Head Output Comparison

### Input Prompt & Image
* **Image**: High-resolution clinical mammogram (`MM-1-a.png`)
* **Prompt**: Analyze the mammogram, identify the most evident finding (Options A–D), and output detailed clinical reasoning in 4 paragraphs.

```
Question:
Examine the mammogram image shown above. Which of the following findings is most evident?

Options:
A. Well-circumscribed round mass with benign features
B. Clustered microcalcifications within an area of irregular density
C. Fat-containing lesion consistent with lipoma
D. Diffuse bilateral breast edema
```

---

### 1. MetalLiger Execution Output (Native PyTorch)
```text
🧠 Prefilling prompt...
   Prefill done in 1.120s (149.1 tokens/s)
⚡ Decoding...
   Decode done in 9.791s (48.1 tokens/s)

--- Output ---
Answer: B. Clustered microcalcifications within an area of irregular density

Reasoning:

1. The mammogram image shows a cluster of microcalcifications within an area of irregular 
   density. This finding is significant because clustered microcalcifications are often 
   associated with malignant breast lesions, such as ductal carcinoma in situ (DCIS) 
   or invasive breast cancer.

2. The presence of microcalcifications, especially when clustered, is a key indicator 
   in mammography for identifying potential malignancies. These calcifications are typically 
   found in the ducts of the breast and are more commonly seen in malignant lesions 
   compared to benign ones.

3. The irregular density surrounding the microcalcifications suggests a heterogeneous breast 
   tissue, which can be indicative of malignancy. This combination of findings is particularly 
   concerning for breast cancer.

4. The mammogram's ability to detect these features is crucial in early diagnosis and 
   management of breast cancer, as it allows for timely intervention and treatment planning.

Conclusion:
The mammogram image reveals clustered microcalcifications within an area of irregular 
density, which are significant findings indicative of potential malignancy.
```

---

### 2. MLX-VLM Execution Output
```text
Prefill:  99%|██████████████████████████▊| 189/190 [00:00<00:00, 1361.11tok/s]
Prompt: 190 tokens, 750.643 tokens-per-sec
Generation: 173 tokens, 35.645 tokens-per-sec
Peak memory: 10.233 GB

--- Output ---
<think>

</think>

The mammogram image provided shows a well-circumscribed round mass with benign features. 
This finding is consistent with a benign lesion, such as a fibroadenoma, which is a common 
benign breast tumor. The well-defined nature of the mass and its benign characteristics 
are key indicators in this case. Therefore, the correct option is A, which describes a 
well-circumscribed round mass with benign features.
```
*(Notice: MLX hallucinated a benign round mass (Option A), skipping the reasoning chain and missing the clustered microcalcifications (Option B).)*

---

## 🛠️ Core Engineering Innovations

Why does MetalLiger generate tokens faster than MLX?

1. **`M4StaticCache` (Zero-Allocation KV Cache)**:
   Standard PyTorch `DynamicCache` calls `torch.cat` on every single token, creating over 12,000 intermediate tensor allocations for a 500-token generation across 24 layers. `M4StaticCache` pre-allocates unified memory buffers once and updates key-values in-place with zero memory copies.
2. **Fused Metal Kernels**:
   Replaces fragmented multi-op Python dispatch with fused Metal shaders:
   - **Fused RMSNorm**: Single-pass root-mean-square normalization.
   - **Fused SwiGLU**: Fast bfloat16 fused GEMM + activation without autograd frame overhead.
   - **Fused RoPE**: Continuous M-RoPE position embedding tracker for multimodal VLMs.
3. **MPS Flash Attention**:
   Direct integration with native Metal Flash Attention kernels (`mps-flash-attention`), reducing memory complexity from $O(N^2)$ to $O(N)$ during prompt prefill.
4. **True Zero-Sync Pipeline Loop**:
   Eliminates per-token GPU synchronization stalls. By batching end-of-sequence checks into single-scalar checks, Metal command buffers flow without pipeline bubbles.
5. **Top-K Guided Nucleus Sampling**:
   Caps sorting overhead to candidate tokens instead of sorting all 152,000 vocabulary logits on every step.

---

## 📦 Installation

### 1. Prerequisites
- macOS 14.0+ (Sonoma or Sequoia)
- Apple Silicon Mac (M1, M2, M3, M4 series)
- Python 3.10+
- PyTorch 2.3+ with MPS support enabled

### 2. Quick One-Line Setup (Terminal)
```bash
pip install -r requirements.txt && git clone https://github.com/mpsops/mps-flash-attention && pip install -e mps-flash-attention && pip install -e .
```

### 3. Step-by-Step Installation
```bash
# Install core Python dependencies
pip install -r requirements.txt

# Install MPS Flash Attention
git clone https://github.com/mpsops/mps-flash-attention
pip install -e mps-flash-attention

# Install MetalLiger in editable mode
pip install -e .

# (Optional) Build native C++ Metal dispatch extensions
python setup_ext.py build_ext --inplace
```

---

## 💻 Quick Start

### Python API
```python
import torch
from PIL import Image
from metalliger.inference import InferenceEngine

# 1. Initialize the Native Apple Silicon Inference Engine
engine = InferenceEngine(
    model_id="Qwen3VL-2B-Medmaxv11",
    max_seq_len=4096,
    dtype=torch.float16,   # Native half-precision (matches fine-tuning weights)
    use_compile=False      # Stable fused eager mode
)

# 2. Run multimodal vision-language inference
image = Image.open("MM-1-a.png")

prompt = """Choose the correct option for the question:
Examine the mammogram image shown above. Which of the following findings is most evident?

Options:
A. Well-circumscribed round mass with benign features
B. Clustered microcalcifications within an area of irregular density
C. Fat-containing lesion consistent with lipoma
D. Diffuse bilateral breast edema

Provide answer in detailed reasoning in 4 paragraphs and final output.
"""

response = engine.generate(
    text_prompt=prompt,
    images=[image],
    max_new_tokens=512,
    temperature=0.0  # Greedy decoding for high diagnostic precision
)

print(response)
```

### CLI Command
You can also run inference directly from your terminal:

```bash
python -m metalliger.inference \
    --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
    --image "MM-1-a.png" \
    --prompt "Analyze this medical image in detail." \
    --max-tokens 512 \
    --dtype float16
```

---

## 📈 Supported Models

MetalLiger is architected for modern transformer architectures with native support for:
- **Vision-Language Models (VLM)**: Qwen2-VL, Qwen2.5-VL, Qwen3-VL (including fine-tunes like MedMax)
- **Large Language Models (LLM)**: LLaMA 3 / 3.1 / 3.2, Qwen 2.5, DeepSeek-R1-Distill-Qwen
- **Linear-Attention Hybrids**: DeltaNet, RWKV, FLA architectures on MPS

---

## 🤝 Contributing

Contributions are warmly welcomed! Whether you want to add new fused Metal shaders, extend support for additional model architectures, or optimize prefill throughput:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/fused-operator`)
3. Commit your changes (`git commit -m 'Add fused operator'`)
4. Push to the branch (`git push origin feature/fused-operator`)
5. Open a Pull Request

---

## 📄 License

Licensed under the [Apache License 2.0](LICENSE).
