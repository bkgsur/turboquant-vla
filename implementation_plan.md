# TurboQuant VLA for Raspberry Pi 4B — Detailed Implementation Plan

## Overview

**TARGET**: HiWonder TurboPi with Raspberry Pi 4B  
**Processor**: Quad-core ARM Cortex-A72 (1.5 GHz)  
**Memory**: 4GB RAM  
**Goal**: Deploy TurboQuant KV cache compression for real-time VLA inference at 5-10 Hz with <5% accuracy loss

**Target models**: 
- Primary: Octo-small (500M params)
- Backup: π₀ quantized (3B params, pre-quantized)

---

## Understanding the Models: Octo-Small, π₀, and TurboQuant

### KV Cache Problem in VLAs

Vision-Language-Action models process two sequences:
1. **Visual tokens** (~400-1000): From image encoder (vision backbone)
2. **Action history tokens** (~20-100 per timestep): Previous actions + state

During autoregressive decoding, the KV cache **grows with every timestep**:
```
Step 1: [vision_tokens] → action_1 (cache: ~1 MB)
Step 2: [vision_tokens] + [prev_actions] → action_2 (cache: ~2 MB)
Step N: Full history (cache: ~37 MB for seq_len=1000)
```

**Memory formula (FP16, 2 bytes per value)**:
```
KV_cache = 2 × seq_len × hidden_dim × 2 bytes
For seq_len=1000, hidden_dim=768 per layer:
Per layer: 3.1 MB → 12 layers = 37 MB total
```

### Octo-Small (500M params)

**Architecture**:
- Vision encoder: ViT-base (frozen, ~86M params) → ~570 visual tokens (hidden_dim=768)
- Transformer decoder: 12 layers, 12 heads, 768 hidden_dim (~400M trainable params)
- Action head: Linear projection → 7D robot action

**On Raspberry Pi 4B without optimization**:
- Model weights: 2 GB (FP16)
- KV cache max: 37 MB (growing with sequence length)
- Activations: 300-400 MB
- **Total: ~2.8 GB** (tight fit, causes slowdown or OOM)

**Inference latency**: 300-400ms per action (2.5-3.3 Hz) ← **Too slow for real-time control**

### π₀ Quantized (3B params, pre-quantized)

**Key differences**:
- Larger base model (3B params) but **weights are INT8 quantized**
- Actual memory footprint: 1-1.5 GB (vs 2 GB for Octo-small)
- Slightly slower inference (~200-300ms) but fits more comfortably in 4GB RAM

**Trade-off**: Larger model but more aggressive weight quantization → tighter memory envelope.

### TurboQuant: The Solution

**Core insight**: Not all KV cache tokens need full precision.

**Strategy**:
```
Recent tokens (last 512):  Keep in FP16 (critical for attention scores)
Historical tokens:        Quantize to INT4 (4x smaller, less critical)

Memory savings:
- FP16 window (512): 786 KB per layer
- INT4 historical (488): 188 KB per layer
- Total: ~975 KB per layer (vs 3 MB) = 3.1x compression
```

**Result**: KV cache shrinks from 37 MB → 12-15 MB per 12-layer model

**Why it works**:
- Recent tokens contribute most to attention weights
- Older tokens have exponentially decaying importance
- INT4 quantization error doesn't accumulate (dequantized on each use)
- Residual window (FP16 recent) preserves accuracy of recent context

### Performance on Raspberry Pi 4B with TurboQuant

```
Memory budget (4 GB):
├── Model weights (Octo-small FP16): 2 GB
├── Activations: 300-400 MB
├── KV cache (TurboQuant): 12-15 MB (fixed, not growing)
├── PyTorch runtime: 100 MB
└── OS overhead: 500 MB
= ~3 GB total (comfortable fit with margin)

Inference latency with TurboQuant:
├── Vision encoding: ~50-100 ms
├── Transformer (with dequantized KV): ~100-150 ms
└── Total: 150-250 ms per action = 4-6.7 Hz ✓ USABLE
```

### Accuracy Impact

Expected accuracy retention with TurboQuant:
- **Baseline (Octo-small)**: 85-90% task success
- **With 4-bit TurboQuant**: 82-88% (-3-5% loss)
- **Why minimal loss**: Recent tokens (FP16) preserve temporal reasoning; older tokens contribute less

### Model Choice for Pi 4B

| Metric | Octo-Small + TurboQuant | π₀ + TurboQuant | Winner |
|--------|------------------------|-----------------|--------|
| **Inference latency** | 200-300 ms (5 Hz) | 180-280 ms (3.6-5.5 Hz) | π₀ slightly faster |
| **Memory used** | 2.3 GB | 1.6 GB | π₀ (more margin) |
| **Accuracy** | ~85% | ~80% (-5% from quant) | Octo-small |
| **Complexity** | Single quantization (KV) | Double quantization (weights+KV) | Octo-small |
| **Recommendation** | **Best choice** (balanced) | Consider if extreme constraints | Octo-small |

**Decision**: Start with **Octo-small + TurboQuant** for Phase 2 validation; validate π₀ in Phase 2.3 if time permits.

---

## Learning Plan: Understanding Models & KV Cache Compression

Before implementing Phase 1, build a clear mental model of how Octo-small, π₀, and TurboQuant work together. This structured learning path ensures you understand *why* we're building what we're building.

### Step 1: VLA Fundamentals
**Goal**: Understand what a Vision-Language-Action model is

**Topics**:
- [ ] What is a VLA? (image + action history → next action)
- [ ] Why robotics? (real-time control, learn from demonstrations)
- [ ] High-level architecture (vision encoder → decoder → action head)
- [ ] Difference from LLMs (action-specific, bounded sequences)

**Questions to answer**:
- [ ] What does "action history" mean?
- [ ] Why separate vision encoding from action decoding?

**Activities**:
- [ ] Read Octo model card on HuggingFace
- [ ] Skim LeRobot repository to see VLA in action

**Checkpoint**: Can you explain "VLAs predict the next robot action based on image and action history"

---

### Step 2: Octo-Small Architecture
**Goal**: Know the exact structure of Octo-small (500M params)

**Topics**:
- [ ] Vision encoder:
  - [ ] Which backbone? (ViT, ResNet?)
  - [ ] How many tokens output?
  - [ ] Hidden dimension?
- [ ] Transformer decoder:
  - [ ] Number of layers?
  - [ ] Number of heads?
  - [ ] Head dimension?
  - [ ] Trainable parameters?
- [ ] Action head (output dimension)
- [ ] Parameter breakdown (total 500M)

**Questions to answer**:
- [ ] What's hidden_dim in Octo-small? *(Answer: 768)*
- [ ] How many attention heads per layer?
- [ ] Total vision tokens from encoder?

**Activities**:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("octo-model-id")
print(model)  # Inspect architecture
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total / 1e6} M")
```

**Checkpoint**: Create architecture diagram with layer counts and dimensions

---

### Step 3: KV Cache Growth Problem
**Goal**: Understand why KV cache explodes and becomes critical for Pi

**Topics**:
- [ ] What is KV cache? (Key and Value embeddings from prior tokens)
- [ ] Why cache? (Avoid recomputing attention for processed tokens)
- [ ] How does it grow in autoregressive decoding?
  ```
  Step 1: [vision] → action (K₁, V₁ cached)
  Step 2: [vision] + [action₁] → action (K₁+K₂, V₁+V₂ cached)
  Step N: Full history cached
  ```
- [ ] Memory calculation:
  - [ ] Per value: 2 bytes (FP16)
  - [ ] Per token: hidden_dim × 2 bytes
  - [ ] Per layer: seq_len × hidden_dim × 2 × 2 bytes (K and V)
  - [ ] All layers: × num_layers (12 for Octo-small)

**Questions to answer**:
- [ ] KV cache size for Octo-small at seq_len=1000? *(Answer: ~37 MB)*
- [ ] At seq_len=2000? *(Answer: ~75 MB)*
- [ ] Why is 75 MB a problem on Pi 4B (4GB)?

**Activities**:
```python
def kv_cache_size_mb(seq_len, hidden_dim=768, num_layers=12):
    bytes = 2 * seq_len * hidden_dim * 2 * num_layers
    return bytes / 1e6

# Octo-small calculation
for seq_len in [100, 500, 1000, 2000]:
    print(f"seq_len={seq_len}: {kv_cache_size_mb(seq_len):.1f} MB")
```

**Checkpoint**: Create graph of KV cache growth vs sequence length

---

### Step 4: TurboQuant Solution
**Goal**: Understand how TurboQuant reduces KV cache and preserves accuracy

**Topics**:
- [ ] Core insight: Not all tokens need full precision
  - [ ] Recent tokens: Critical (keep FP16)
  - [ ] Old tokens: Less important (quantize to INT4/INT3)
- [ ] Quantization mechanics:
  - [ ] INT4: 4 bits per value (vs 16 bits FP16) = 4x compression
  - [ ] Formula: `q = round((x - min) / scale)`
  - [ ] Dequantization: `x_approx = q × scale + min`
- [ ] Residual window strategy:
  - [ ] Keep last N tokens in FP16 (e.g., 512)
  - [ ] Quantize older tokens to INT4
  - [ ] On each step: new token joins FP16, oldest FP16 token moves to INT4
- [ ] Memory savings:
  ```
  Without: 2 × 1000 × 768 × 2 × 12 = 37.3 MB
  With (window=512, INT4): ~12-15 MB (3.2x compression)
  ```
- [ ] Why accuracy preserved:
  - [ ] Recent tokens (FP16) capture latest context
  - [ ] Old tokens have exponentially decaying attention weights
  - [ ] INT4 error doesn't accumulate (dequantized on each use)

**Questions to answer**:
- [ ] What does INT4 mean? *(Answer: 4 bits per value, 0.25× size)*
- [ ] Residual window of 512 means what? *(Answer: last 512 tokens FP16, rest INT4)*
- [ ] Why does this preserve accuracy? *(Answer: recent tokens most important for attention)*

**Activities**:
```python
# Simulate quantized KV cache
class QuantizedKVCache:
    def __init__(self, hidden_dim=768, window_size=512):
        self.fp16_window = []      # Recent tokens (full precision)
        self.int4_history = []      # Old tokens (quantized)
    
    def add_token(self, k, v):
        self.fp16_window.append((k, v))
        if len(self.fp16_window) > 512:
            # Move oldest from FP16 to INT4
            self.int4_history.append(quantize(self.fp16_window.pop(0)))
    
    def memory_mb(self):
        fp16_size = len(self.fp16_window) * 768 * 2
        int4_size = len(self.int4_history) * 768 * 0.5
        return (fp16_size + int4_size) / 1e6
```

**Checkpoint**: Calculate KV cache size with TurboQuant for different window sizes

---

### Step 5: π₀ Quantized Model
**Goal**: Know how π₀ differs and when to use it

**Topics**:
- [ ] π₀ overview:
  - [ ] Base: 3B parameters (larger than Octo-small 500M)
  - [ ] **But**: Model weights already INT8 quantized
  - [ ] Result: ~1-1.5 GB memory (vs 2 GB for Octo-small FP16)
- [ ] Double quantization:
  - [ ] Model weights: INT8 (π₀ baseline)
  - [ ] KV cache: INT4 via TurboQuant
  - [ ] Combined: Very aggressive compression
- [ ] Trade-offs:
  - [ ] Pros: Smallest memory footprint
  - [ ] Cons: Double quantization may hurt accuracy, slower inference
- [ ] When to use:
  - [ ] Octo-small: Balanced (accuracy + speed)
  - [ ] π₀: Maximum memory efficiency

**Questions to answer**:
- [ ] Why is π₀ called "3B" if it's smaller? *(Answer: 3B is raw params; INT8 compresses 8×)*
- [ ] Which model to prioritize? *(Answer: Octo-small first; π₀ as backup)*

**Activities**:
- [ ] Check HuggingFace for π₀ model availability
- [ ] Compare model cards: Octo-small vs π₀

**Checkpoint**: Create comparison table (size, speed, accuracy)

---

### Learning Validation Checklist

After completing all 5 steps, you should answer these:

**VLA Basics**:
- [ ] What input does a VLA take?
- [ ] What output does it produce?
- [ ] Why useful for robotics?

**Octo-Small**:
- [ ] Total parameters? *(500M)*
- [ ] Hidden dimension? *(768)*
- [ ] Transformer layers? *(12)*
- [ ] Vision tokens? *(~570)*

**KV Cache Problem**:
- [ ] What is KV cache?
- [ ] Why cache it?
- [ ] Cache size at seq_len=1000? *(~37 MB)*
- [ ] Why problem on Pi? *(Limited RAM, causes slowdown/OOM)*

**TurboQuant**:
- [ ] What is INT4 quantization? *(4 bits per value, 0.25× size)*
- [ ] What is residual window? *(Recent tokens FP16, old tokens INT4)*
- [ ] Compression ratio? *(3-4x)*
- [ ] Why accuracy preserved? *(Recent tokens FP16, old tokens less important)*

**π₀ Model**:
- [ ] How big is π₀? *(~1-1.5 GB with INT8)*
- [ ] Difference from Octo-small? *(Pre-quantized weights)*
- [ ] When use π₀? *(Maximum memory efficiency)*

---

### Deliverables After Learning Plan

Create these before moving to Phase 1 implementation:

1. **Architecture Diagram**: Octo-small with layer counts, hidden dims, token flows
2. **KV Cache Graph**: Memory vs sequence length (with and without TurboQuant)
3. **Quantization Explanation**: Written summary of how residual window works
4. **Model Comparison**: Table comparing Octo-small vs π₀ vs combinations
5. **Quick Reference Card**: One-page summary of numbers and formulas

---

## Phase 1: Research & Standalone Library

### 1.1 Study TurboQuant Architecture

**Objective**: Understand KV cache quantization mechanics and implementation details.

**Tasks**:
- [ ] Read arXiv:2504.19874 (TurboQuant paper) — focus on:
  - KV quantization algorithm (per-channel, per-token, residual windows)
  - Calibration procedure for quantization parameters
  - Accuracy trade-offs at 3-bit vs 4-bit
  - Memory savings formula: (seq_len * hidden_dim * bits) vs FP16
- [ ] Review SGLang PR #21617 implementation:
  - How quantization is integrated into attention kernel
  - Calibration data requirements
  - Configuration parameters (window size, bit-width, per-channel vs shared)
- [ ] Review PyPI `turboquant` package:
  - Available APIs and configuration options
  - Supported model formats (HuggingFace, ONNX, TorchScript)
  - Known limitations on ARM platforms

**Deliverables**:
- [ ] Design doc: `docs/turboquant_architecture.md` (500 words)
  - KV quantization algorithm summary
  - Configuration space (bit-width, window size, per-channel)
  - Expected memory savings for Octo-small + π₀
- [ ] Comparison matrix: SGLang vs turboquant PyPI vs custom implementation (pros/cons)

**Success Criteria**: Clear understanding of what needs to be implemented vs what can be reused.

---

### 1.2 Analyze Existing Implementations

**Objective**: Determine what to build vs reuse.

**Tasks**:
- [ ] Clone SGLang repo, locate TurboQuant code:
  - `sglang/srt/layers/kv_cache.py` (or similar)
  - Identify attention kernel modifications
  - Check if optimized for inference (not training)
- [ ] Install and test `turboquant` package:
  - Can it quantize Octo-small out-of-the-box?
  - Does it support ONNX export for ARM deployment?
  - Test on RTX 4050: benchmark memory reduction and latency
- [ ] Research LeRobot/Octo model structure:
  - How is KV cache managed in training/inference?
  - Model size and parameter breakdown (vision encoder, language model, action head)
  - Inference API and input/output formats

**Tasks**:
- [ ] Clone and setup:
  - `git clone https://github.com/sgl-project/sglang.git`
  - `pip install turboquant` (or build from source if needed)
  - `git clone https://github.com/huggingface/lerobot.git`
- [ ] Test scripts:
  - `scripts/test_turboquant_existing.py` — load Octo model, apply turboquant, measure memory
  - `scripts/benchmark_existing.py` — latency on RTX 4050 for different bit-widths
- [ ] Document findings in `docs/implementation_options.md`

**Success Criteria**: Decision made on whether to use turboquant package directly or build custom quantizer.

---

### 1.3 Build Standalone TurboQuant KV Cache Library

**Objective**: Create a reusable, Pi-compatible KV cache quantizer.

**Architecture**:
```
turboquant_kv/
├── __init__.py
├── quantizer.py          # Core quantization logic
├── calibration.py        # Calibration data collection
├── kernels/
│   ├── __init__.py
│   └── quantized_attention.py  # Optimized attention with quantized KV
├── config.py             # Configuration dataclass
├── utils.py              # Helper functions
└── tests/
    ├── test_quantizer.py
    └── test_accuracy.py
```

**Tasks**:

**1.3.1 Core Quantization Module (`quantizer.py`)**
- [ ] Implement `KVQuantizer` class:
  ```python
  class KVQuantizer:
      def __init__(self, bit_width=4, per_channel=True, window_size=512):
          self.bit_width = bit_width
          self.per_channel = per_channel
          self.window_size = window_size  # residual window for unquantized KV
      
      def quantize_kv(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Dict]:
          """Quantize K and V, return quantized tensors + metadata (scale, zero_point)"""
      
      def dequantize_kv(self, k_q: Tensor, v_q: Tensor, scales: Dict, zero_points: Dict) -> Tuple[Tensor, Tensor]:
          """Restore to FP16 for attention computation"""
      
      def compute_scaling_params(self, k: Tensor, v: Tensor) -> Tuple[Dict, Dict]:
          """Compute per-channel or per-token scales and zero-points"""
  ```
- [ ] Support quantization strategies:
  - Per-channel (separate scale per hidden dim)
  - Per-token (separate scale per sequence position)
  - Symmetric (zero-point = 0) vs asymmetric
- [ ] Implement residual window: keep last `window_size` tokens in FP16, quantize rest

**1.3.2 Calibration Module (`calibration.py`)**
- [ ] `Calibrator` class:
  ```python
  class Calibrator:
      def collect_stats(self, model, dataloader, num_batches=100) -> Dict:
          """Run inference, collect K/V statistics (min, max, distribution)"""
      
      def compute_quantization_params(self, stats: Dict) -> Tuple[Dict, Dict]:
          """Convert stats to scales and zero-points"""
  ```
- [ ] Calibration on small Octo dataset (100-200 inference examples)

**1.3.3 Optimized Attention Kernel (`kernels/quantized_attention.py`)**
- [ ] `QuantizedAttention` module:
  ```python
  class QuantizedAttention(nn.Module):
      def __init__(self, hidden_dim, quantizer: KVQuantizer):
          self.quantizer = quantizer
          self.attention = standard_scaled_dot_product_attention  # from PyTorch
      
      def forward(self, q, k, v, kv_cache=None):
          """Attention with quantized K/V cache"""
          if kv_cache is not None:
              k_cached_q, v_cached_q, metadata = kv_cache
              # Dequantize, append new k/v, requantize
              k_full = dequantize + cat + quantize
              v_full = dequantize + cat + quantize
          return attention(q, k_full, v_full)
  ```
- [ ] Backward pass: gradient computation through quantization (straight-through or learned)

**1.3.4 Configuration (`config.py`)**
- [ ] `QuantizerConfig` dataclass:
  ```python
  @dataclass
  class QuantizerConfig:
      bit_width: int = 4  # 3, 4, 8
      per_channel: bool = True
      window_size: int = 512  # unquantized residual
      symmetric: bool = True
      calibration_samples: int = 100
  ```

**1.3.5 Testing**
- [ ] Unit tests:
  - [ ] `test_quantizer.py`: Quantization round-trip (quantize → dequantize → close to original)
  - [ ] Test different bit-widths, strategies
  - [ ] Measure memory savings: (k_q, v_q) size < (k, v) * (bit_width / 16)
- [ ] Accuracy tests:
  - [ ] `test_accuracy.py`: Inference on Octo test set, compare quantized vs FP16 outputs
  - [ ] Action prediction MSE, action sequence accuracy
  - [ ] Tolerance: < 5% action accuracy drop

**Deliverables**:
- [ ] PyPI-publishable package: `turboquant_kv/`
- [ ] Unit tests: > 90% code coverage
- [ ] Benchmark on RTX 4050:
  - Memory reduction: target 4x for KV cache
  - Latency overhead: < 10% (dequant + attention still faster than FP16)
- [ ] README with example usage

**Success Criteria**:
- Standalone library works on PyTorch CPU/GPU
- Quantization + dequantization cycle < 5ms on RTX 4050
- Memory savings meet 4x target
- Accuracy loss < 5% on Octo

---

## Phase 2: Model Integration & Pi Validation

### 2.1 Integrate with Octo-Small Model

**Objective**: Apply TurboQuant to Octo-small inference pipeline.

**Tasks**:
- [ ] Load Octo-small from HuggingFace:
  ```python
  from octo.model import OctoModel
  model = OctoModel.from_pretrained("octo-base-1B")  # or octo-small if available
  ```
- [ ] Wrap KV cache operations with TurboQuant:
  - Identify where KV cache is stored/updated in Octo
  - Replace cache management with quantized version
  - Ensure autoregressive decoding still works (append new K/V, requantize incrementally)
- [ ] Create `OctoQuantizedInference` module:
  ```python
  class OctoQuantizedInference:
      def __init__(self, model_id="octo-base-1B", quantizer_config: QuantizerConfig):
          self.model = OctoModel.from_pretrained(model_id)
          self.quantizer = KVQuantizer(quantizer_config)
          self._calibrate_on_dataset()  # 100 examples
      
      def infer_action(self, image, history) -> np.ndarray:
          """Single step action prediction with quantized KV cache"""
          with torch.no_grad():
              output = self.model(image, history, kv_cache_quantizer=self.quantizer)
          return output.action.cpu().numpy()
  ```
- [ ] Test on RTX 4050 with Octo evaluation dataset:
  - Latency: measure per-action inference time
  - Accuracy: action MSE vs baseline
  - Memory: peak GPU memory during inference

**Deliverables**:
- [ ] `octo_quantized_inference.py` — integration module
- [ ] Benchmark script: `scripts/benchmark_octo_quantized.py`
  - Latency vs baseline (FP16)
  - Accuracy drop (action MSE)
  - Memory savings

**Success Criteria**:
- Octo-small + TurboQuant runs on RTX 4050
- Memory < 3GB (can fit on Pi with margin)
- Latency < 200ms per action (5 Hz minimum)
- Action accuracy loss < 5%

---

### 2.2 Test on Actual Raspberry Pi 4B

**Objective**: Validate quantization strategy works on ARM hardware.

**Setup**:
- [ ] Flash Pi 4B with 64-bit Raspberry Pi OS (bullseye or later for better ARM support)
- [ ] Install dependencies:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install transformers huggingface_hub
  pip install onnxruntime-rpi  # or standard onnxruntime if ARM build available
  ```
- [ ] SSH into Pi for remote development

**Tasks**:
- [ ] Port TurboQuant library to Pi:
  - [ ] Install `turboquant_kv` package on Pi
  - [ ] Test quantization/dequantization on CPU
  - [ ] Verify memory usage: quantized KV << unquantized
- [ ] Benchmark Octo-small on Pi:
  - [ ] Create lightweight test script: load model, run 10 inferences
  - [ ] Measure:
    - Model loading time
    - Per-action inference latency (target: < 500ms for 2-5 Hz)
    - Peak memory usage
    - Thermal behavior (Pi throttling?)
  - [ ] Script: `scripts/benchmark_octo_pi.py`
- [ ] Test accuracy on Pi:
  - [ ] Run Octo evaluation on Pi with quantized model
  - [ ] Compare to RTX 4050 baseline
  - [ ] Document any differences (might be due to FP32 vs FP16 precision)

**Challenges & Mitigations**:
- **Challenge**: Model doesn't fit in 4GB RAM
  - *Mitigation*: Model streaming (load layer-by-layer), aggressive quantization (3-bit), weight quantization too
- **Challenge**: Inference too slow (> 1 second per action)
  - *Mitigation*: Profile on Pi, optimize hotspots in TorchScript, consider ONNX export
- **Challenge**: Thermal throttling during inference
  - *Mitigation*: Heatsink, fan, or reduce batch inference frequency

**Deliverables**:
- [ ] Benchmark results on Pi: `docs/pi_benchmark_results.md`
  - Model loading time
  - Latency: mean, min, max, std over 100 inferences
  - Memory: peak, average
  - Thermal: CPU temp during inference
- [ ] Comparison: Octo-small + FP16 vs + 4-bit quantization vs + 3-bit quantization

**Success Criteria**:
- Octo-small + 4-bit quantization runs on Pi 4B without OOM
- Inference latency 5-10 Hz (100-200ms per action)
- Memory footprint < 3GB
- Accuracy loss < 5%

---

### 2.3 Optimize for π₀ (Optional, Advanced)

**Objective**: If time permits, validate on π₀ quantized model (even smaller).

**Tasks**:
- [ ] Load π₀ quantized variant (if available on HuggingFace)
- [ ] Apply TurboQuant on top (quantized model + quantized KV)
- [ ] Benchmark on Pi: latency, memory, accuracy
- [ ] Document findings

---

## Phase 3: ONNX Export & ARM Optimization

### 3.1 Export Quantized Model to ONNX

**Objective**: Create ONNX runtime version for faster ARM inference.

**Tasks**:
- [ ] Export Octo-small to ONNX:
  ```python
  import torch
  import torch.onnx
  
  model = OctoModel.from_pretrained(...)
  dummy_input = (torch.randn(...), torch.randn(...))  # image, history
  torch.onnx.export(model, dummy_input, "octo_quantized.onnx", ...)
  ```
- [ ] Test ONNX model on RTX 4050:
  - Load with ONNX Runtime (CPU + GPU)
  - Verify outputs match PyTorch
  - Measure latency
- [ ] Quantize ONNX model weights (optional):
  - Use ONNX quantization tools (QDQ format)
  - 8-bit weight quantization on top of KV quantization
  - Test accuracy impact

**Deliverables**:
- [ ] `octo_quantized.onnx` — quantized ONNX model
- [ ] ONNX test script: `scripts/test_onnx_inference.py`
- [ ] Comparison: PyTorch vs ONNX latency/memory

---

### 3.2 Optimize for ARM Runtime (Raspberry Pi)

**Objective**: Achieve best possible latency on ARM Cortex-A72.

**Tasks**:
- [ ] Profile inference on Pi:
  - [ ] Use `cProfile` or ARM Streamline to identify bottlenecks
  - [ ] Likely hotspots: matrix multiply, softmax, layer norm
- [ ] Optimization strategies:
  - [ ] Use ONNX Runtime with ARM optimizations:
    ```python
    import onnxruntime as ort
    session_options = ort.SessionOptions()
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.add_session_config_entry("optimization.level", "all")
    sess = ort.InferenceSession("octo_quantized.onnx", session_options)
    ```
  - [ ] Reduce model size further if needed:
    - Prune low-weight connections
    - Distill into smaller model (if time permits)
  - [ ] Batch inference? (if robot allows 1-5 frame batch)
- [ ] Benchmark after optimization:
  - [ ] Latency breakdown by layer
  - [ ] Peak memory during inference
  - [ ] Throughput: max actions per second

**Deliverables**:
- [ ] Optimized ONNX model + ONNX Runtime config
- [ ] Profiling report: `docs/pi_profiling_report.md`
  - Latency breakdown
  - Memory timeline
- [ ] Updated benchmark: `scripts/benchmark_octo_pi_optimized.py`

**Success Criteria**:
- Inference latency: 5-10 Hz (100-200ms per action)
- Peak memory < 3GB
- Thermal: no throttling under sustained inference

---

## Phase 4: LeRobot Integration & Robot Deployment

### 4.1 Integrate with LeRobot Inference Pipeline

**Objective**: Make quantized inference a drop-in replacement in LeRobot.

**Tasks**:
- [ ] Study LeRobot inference API:
  - Where is model inference called?
  - How is KV cache managed?
  - What's the expected input/output format?
- [ ] Create LeRobot-compatible inference wrapper:
  ```python
  class LeRobotQuantizedPolicy:
      """Drop-in replacement for LeRobot policy with TurboQuant KV cache"""
      def __init__(self, model_id="octo-base-1B", quantizer_config=None):
          self.policy = LeRobotPolicy.from_pretrained(model_id)
          self.quantizer = KVQuantizer(quantizer_config or QuantizerConfig())
      
      def forward(self, observation) -> Dict[str, Tensor]:
          """Inference with quantized KV cache"""
          # observation: {image: Tensor, state: Tensor, ...}
          action = self.policy(observation, kv_cache_quantizer=self.quantizer)
          return action
  ```
- [ ] Test with LeRobot demo dataset
- [ ] Document how to use quantized policy

**Deliverables**:
- [ ] `lerobot_quantized_policy.py` — wrapper class
- [ ] Integration docs: `docs/lerobot_integration.md`
- [ ] Example notebook: `examples/octo_lerobot_inference.ipynb`

---

### 4.2 End-to-End Testing on TurboPi

**Objective**: Validate full pipeline on actual robot.

**Setup**:
- [ ] Install LeRobot and dependencies on Pi
- [ ] Connect TurboPi camera and arm servos
- [ ] Sync quantized policy weights to Pi

**Tasks**:
- [ ] Load policy and run closed-loop control:
  ```python
  from lerobot_quantized_policy import LeRobotQuantizedPolicy
  
  policy = LeRobotQuantizedPolicy("octo-base-1B", quantizer_config)
  
  for frame in camera_stream:
      action = policy({"image": frame})
      arm.execute_action(action)  # move servos
      time.sleep(1.0 / TARGET_HZ)  # maintain 5-10 Hz
  ```
- [ ] Collect metrics:
  - [ ] Inference latency: per-action time
  - [ ] Action accuracy: compare to ground-truth demonstrations
  - [ ] Throughput: sustained inference frequency
  - [ ] Thermal: CPU temp, throttling events
  - [ ] Memory: peak usage during inference
- [ ] Run test task (e.g., pick & place, object following)
- [ ] Compare quantized vs baseline (if baseline fits on Pi)

**Deliverables**:
- [ ] Test results: `docs/turbopi_test_results.md`
  - Inference latency, memory, thermal
  - Task success rate (if applicable)
  - Comparison quantized vs baseline
- [ ] Demo video/script: `examples/turbopi_demo.py`
- [ ] Deployment guide: `docs/turbopi_deployment.md`

**Success Criteria**:
- Inference at 5-10 Hz on TurboPi
- Quantized model fits in 4GB RAM
- Task execution successful (if task-specific)
- Thermal stable (no throttling after warmup)

---

## Phase 5: Documentation & Polish

### 5.1 Code Documentation

- [ ] Docstrings: all public classes/functions
- [ ] Type hints: all functions
- [ ] README: installation, quick start, API reference
- [ ] Examples: simple inference, custom calibration, deployment

### 5.2 Performance Documentation

- [ ] Benchmark suite: `scripts/run_all_benchmarks.sh`
- [ ] Performance guide: `docs/performance_guide.md`
  - How quantization affects accuracy/speed trade-off
  - Tips for custom models
  - Troubleshooting low memory / slow inference

### 5.3 Deployment Guide

- [ ] Step-by-step: setup Pi, install packages, run policy
- [ ] Configuration: how to tune quantizer for custom models
- [ ] Troubleshooting: common issues (OOM, slow, throttling)

---

## File Structure

```
turboquant-vla/
├── README.md                          # Overview, quick start
├── plan.md                            # High-level goals
├── implementation_plan.md             # This file
│
├── turboquant_kv/                     # Core library
│   ├── __init__.py
│   ├── quantizer.py                   # KVQuantizer class
│   ├── calibration.py                 # Calibration logic
│   ├── config.py                      # QuantizerConfig
│   ├── utils.py                       # Helpers
│   ├── kernels/
│   │   ├── __init__.py
│   │   └── quantized_attention.py     # Optimized attention
│   └── tests/
│       ├── test_quantizer.py
│       ├── test_accuracy.py
│       └── fixtures/                  # Test data
│
├── octo_integration/                  # Octo model integration
│   ├── __init__.py
│   ├── octo_quantized.py              # OctoQuantizedInference
│   └── tests/
│       └── test_octo_quantized.py
│
├── lerobot_integration/               # LeRobot integration
│   ├── __init__.py
│   ├── lerobot_policy.py              # LeRobotQuantizedPolicy
│   └── tests/
│       └── test_lerobot_policy.py
│
├── scripts/
│   ├── benchmark_octo_quantized.py    # RTX 4050 benchmarks
│   ├── benchmark_octo_pi.py           # Pi benchmarks
│   ├── benchmark_octo_pi_optimized.py # Optimized Pi benchmarks
│   ├── turbopi_demo.py                # On-robot demo
│   └── run_all_benchmarks.sh          # Benchmark suite
│
├── examples/
│   ├── octo_lerobot_inference.ipynb   # Notebook demo
│   └── turbopi_demo.py                # Robot integration example
│
├── docs/
│   ├── turboquant_architecture.md     # Algorithm overview
│   ├── implementation_options.md      # Decision log
│   ├── pi_benchmark_results.md        # Phase 2 results
│   ├── pi_profiling_report.md         # Phase 3 profiling
│   ├── turbopi_test_results.md        # Phase 4 test results
│   ├── turbopi_deployment.md          # Deployment guide
│   ├── performance_guide.md           # Tuning guide
│   └── troubleshooting.md             # Common issues
│
├── tests/
│   ├── integration_test.py            # End-to-end tests
│   └── accuracy_test.py               # Accuracy validation
│
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
└── .gitignore
```

---

## Phase Milestones

| Phase | Key Deliverables | Success Criteria |
|-------|-----------------|------------------|
| Phase 1 | TurboQuant library, unit tests, RTX 4050 benchmarks | Standalone library works, 4x KV memory reduction |
| Phase 2 | Octo integration, Pi validation, accuracy tests | Runs on Pi 4B, 5-10 Hz, < 5% accuracy loss |
| Phase 3 | ONNX export, ARM optimization, profiling | ONNX model optimized for ARM, profiling report |
| Phase 4 | LeRobot integration, end-to-end testing | Full pipeline on TurboPi, task execution validated |
| Phase 5 | Documentation, deployment guide, polish | Complete docs, reproducible deployment |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| TurboQuant not available for custom models | Medium | High | Start with existing implementations (SGLang, turboquant pkg) |
| 4GB RAM insufficient | Medium | High | 3-bit quantization, weight quantization, model streaming |
| Inference too slow on Pi | Medium | High | Profile early, ONNX export, SIMD optimizations |
| Accuracy drops > 5% | Low | High | Careful calibration, tune window size, per-channel scales |
| Thermal throttling on Pi | Medium | Medium | Heatsink, reduce inference batch, thermal profiling |
| LeRobot API not compatible | Low | Medium | Wrap inference, fork if needed, document clearly |

---

## Success Metrics (End of Phase 5)

- ✅ TurboQuant KV cache library: standalone, tested, documented
- ✅ Octo-small runs on Pi 4B at 5-10 Hz
- ✅ Memory footprint: 2.5-3.5 GB (fits in 4GB with margin)
- ✅ Accuracy loss: < 5% on action prediction
- ✅ Deployment: documented, reproducible
- ✅ Code: tested (> 90% coverage), type-hinted, documented

---

## Future Work (Post-MVP)

- [ ] Weight quantization (8-bit) on top of KV quantization
- [ ] Distillation: train smaller model with quantized KV cache
- [ ] Support π₀ and other VLA models
- [ ] ROS 2 wrapper for broader robot integration
- [ ] FPGA or GPU acceleration (if Jetson deployment added)
- [ ] Multi-token prediction (speculative decoding with quantized cache)
