# Learning Plan: Octo-Small, π₀, and KV Cache Compression with TurboQuant

**Target Hardware**: Raspberry Pi 4B (Quad-core ARM Cortex-A72, 4GB RAM) running on HiWonder TurboPi

## Goal
Build a clear mental model of:
1. How Octo-small and π₀ models work
2. Why KV cache becomes a bottleneck in these models
3. How TurboQuant solves the KV cache problem for ARM Cortex-A72
4. Why this is critical for deploying VLAs on Pi 4B's 4GB RAM with 5-10 Hz latency

---

## Why Pi 4B Matters

**Raspberry Pi 4B Constraints**:
- **4GB RAM** (shared with OS, system processes) → ~2-3 GB available for ML
- **No GPU** → All compute on ARM Cortex-A72 CPU (1.5 GHz)
- **Thermal limits** → Throttles at >80°C
- **Cost** → $75 (vs $800 for Jetson) — enables mass deployment

**Without TurboQuant**:
- Octo-small (FP16 weights): 2 GB + KV cache grows → Out of memory or <1 Hz
- Unusable for real-time robotics (need 5-10 Hz minimum)

**With TurboQuant**:
- Fits in 4GB total RAM with margin for OS overhead
- KV cache fixed at 12-15 MB (doesn't explode with sequence length)
- Achieves 5-10 Hz inference on ARM Cortex-A72

---

## Learning Path (5 Steps)

### Step 1: Understand Vision-Language-Action (VLA) Models
**Duration**: 2-3 hours  
**Goal**: Know what a VLA is and how it processes information

**Topics to understand**:
- [ ] What is a Vision-Language-Action model?
  - Input: Image + action history
  - Output: Next action prediction
  - Use case: Robotics manipulation
- [ ] Architecture pattern:
  - Vision encoder (frozen, extracts image features)
  - Language/action decoder (processes features + history)
  - Action head (outputs robot commands)
- [ ] Difference from LLMs:
  - LLMs: text → text (unbounded sequence)
  - VLAs: image + history → action (bounded, robotics-specific)

**Questions to answer**:
- [ ] What does "action history" mean in a VLA?
- [ ] Why is vision encoding separate from action decoding?
- [ ] How many tokens come from vision vs action history?

**Resources**:
- [ ] Read: Octo model paper (if available) or HuggingFace model card
- [ ] Search: "Vision Language Action models robotics" 
- [ ] Look at: LeRobot repository structure to understand workflow

**Validation checkpoint**:
- [ ] Can you explain: "VLAs predict the next robot action based on image and action history"
- [ ] Can you draw: Vision encoder → Decoder → Action head pipeline

---

### Step 2: Deep Dive into Octo-Small Architecture
**Duration**: 3-4 hours  
**Goal**: Know the exact structure of Octo-small (500M params)

**Topics to understand**:
- [ ] Vision encoder details:
  - [ ] Which backbone? (ViT, ResNet, etc.)
  - [ ] How many tokens does it output?
  - [ ] What's the hidden dimension?
- [ ] Transformer decoder details:
  - [ ] How many layers?
  - [ ] How many attention heads?
  - [ ] What's the attention head dimension?
  - [ ] Total trainable parameters?
- [ ] Action head:
  - [ ] How many output dimensions? (7D for arm pose + gripper)
  - [ ] Any post-processing?
- [ ] Parameter breakdown:
  - [ ] Vision encoder params: ?
  - [ ] Decoder params: ?
  - [ ] Action head params: ?
  - [ ] Total: 500M ✓

**Questions to answer**:
- [ ] In Octo-small, what's the hidden dimension of the transformer?
- [ ] How many attention heads in each transformer layer?
- [ ] If vision outputs 576 tokens at 768 dims, what's the memory of that?

**Resources**:
- [ ] Load Octo-small model and inspect:
  ```python
  from huggingface_hub import from_pretrained
  model = from_pretrained("...")
  print(model)  # Print model structure
  ```
- [ ] Create a simple notebook to profile the model:
  ```python
  # Count parameters
  total = sum(p.numel() for p in model.parameters())
  print(f"Total: {total / 1e6} M params")
  ```

**Validation checkpoint**:
- [ ] Create a diagram: Octo-small architecture with layer counts and dimensions
- [ ] Can you list: vision tokens, hidden dims, num heads, total params

---

### Step 3: Understand KV Cache Growth (The Problem)
**Duration**: 2-3 hours  
**Goal**: Know why KV cache explodes and becomes a problem

**Topics to understand**:
- [ ] What is KV cache in transformers?
  - [ ] K = Key cache (embeddings from previous tokens)
  - [ ] V = Value cache (embeddings from previous tokens)
  - [ ] Why cache it? (Avoid recomputing attention for previous tokens)
- [ ] How does KV cache grow in autoregressive decoding?
  - [ ] Step 1: Process image → K₁, V₁ cached
  - [ ] Step 2: Process image + actions → append K₂, V₂ to cache
  - [ ] Step N: Cache size = sum of all K_i, V_i
- [ ] Memory calculation:
  - [ ] K cache size = seq_len × hidden_dim × bytes_per_value
  - [ ] V cache size = seq_len × hidden_dim × bytes_per_value
  - [ ] Per layer: 2 × seq_len × hidden_dim × 2 bytes (FP16)
  - [ ] All layers: 2 × seq_len × hidden_dim × 2 × num_layers
- [ ] Why it's a problem on Raspberry Pi:
  - [ ] Limited RAM (4 GB)
  - [ ] Growing cache + model weights → OOM
  - [ ] Inference slows down (memory swaps)

**Questions to answer**:
- [ ] For Octo-small (hidden_dim=768, 12 layers), what's the KV cache size at seq_len=1000?
  - Answer: 2 × 1000 × 768 × 2 × 12 = **37.3 MB**
- [ ] If sequence grows to 2000 tokens, what's the new KV cache size?
  - Answer: 2 × 2000 × 768 × 2 × 12 = **74.6 MB**
- [ ] Why is this a problem on Pi 4B (4GB)?
  - Model: 2GB, Activations: 400MB, KV: 75MB → 2.5 GB used, but with swapping/overhead it's tight

**Resources**:
- [ ] Write a calculation script:
  ```python
  def kv_cache_size(seq_len, hidden_dim, num_layers, bytes_per_value=2):
      return 2 * seq_len * hidden_dim * bytes_per_value * num_layers
  
  # Octo-small: hidden=768, layers=12
  for seq_len in [100, 500, 1000, 2000]:
      size_mb = kv_cache_size(seq_len, 768, 12) / 1e6
      print(f"seq_len={seq_len}: {size_mb:.1f} MB")
  ```
- [ ] Profile actual model:
  ```python
  import torch
  from contextlib import nullcontext as ctx_mgr
  
  model.eval()
  with torch.no_grad():
      for seq_len in [100, 500, 1000]:
          input = torch.randn(1, seq_len, 768)  # batch=1, seq=variable
          output = model(input)  # Capture peak memory usage
          # Use torch.cuda.memory_allocated() or system monitor
  ```

**Validation checkpoint**:
- [ ] Create a graph: KV cache size vs sequence length
- [ ] Calculate: Octo-small KV cache at seq_len=500, 1000, 2000
- [ ] Explain: Why growing KV cache causes inference slowdown on Pi

---

### Step 4: Understand KV Cache Quantization (TurboQuant)
**Duration**: 3-4 hours  
**Goal**: Know how TurboQuant reduces KV cache and preserves accuracy

**Topics to understand**:
- [ ] Core insight: Not all tokens need full precision
  - [ ] Recent tokens: Critical for attention (keep FP16)
  - [ ] Old tokens: Less important (can quantize to INT4/INT3)
- [ ] Quantization mechanics:
  - [ ] What does INT4 mean? (4 bits per value instead of 16)
  - [ ] Quantization formula: `q = round((x - min) / scale)`
  - [ ] Dequantization: `x_approx = q × scale + min`
  - [ ] Compression: 4-bit = 0.25× original size (4x compression)
- [ ] Residual window strategy:
  - [ ] Keep last N tokens in FP16 (e.g., N=512)
  - [ ] Quantize tokens before window to INT4
  - [ ] On each step: append new token to FP16, push old token to INT4
- [ ] Memory savings calculation:
  - [ ] Without quantization (seq_len=1000, hidden_dim=768, 12 layers):
    - Per layer: 2 × 1000 × 768 × 2 bytes = 3.1 MB
    - All layers: 37.3 MB
  - [ ] With TurboQuant (window=512, INT4 for old):
    - Recent (512 FP16): 512 × 768 × 2 bytes = 786 KB
    - Old (488 INT4): 488 × 768 × 0.5 bytes = 188 KB
    - Per layer: 974 KB
    - All layers: ~11.7 MB (3.2x compression)
- [ ] Why accuracy is preserved:
  - [ ] Recent tokens (FP16) capture latest context
  - [ ] Old tokens have exponentially decaying attention weights
  - [ ] INT4 error doesn't accumulate (dequantized each use)

**Questions to answer**:
- [ ] What's the difference between INT4 and FP16 in terms of bits? (4 vs 16)
- [ ] For seq_len=1000, what's the KV cache with 4-bit quantization + 512-token window?
  - Answer: ~12-15 MB (vs 37 MB without)
- [ ] Why does keeping the "residual window" in FP16 preserve accuracy?
  - Answer: Recent tokens most important for attention; attention weights decay exponentially for old tokens
- [ ] What happens during inference when we add a new token?
  - Answer: New token joins FP16 window; old token from window gets quantized; quantized cache updated

**Resources**:
- [ ] Read TurboQuant paper (arXiv:2504.19874):
  - Section on residual windows
  - Quantization calibration procedure
  - Accuracy results
- [ ] Create a simulation:
  ```python
  import numpy as np
  
  # Simulate KV cache with quantization
  class QuantizedKVCache:
      def __init__(self, hidden_dim=768, window_size=512, bit_width=4):
          self.hidden_dim = hidden_dim
          self.window_size = window_size
          self.bit_width = bit_width
          self.fp16_window = []  # Recent tokens
          self.int4_cache = []   # Old tokens
      
      def add_token(self, k_token, v_token):
          # Add to FP16 window
          self.fp16_window.append((k_token, v_token))
          
          # If window full, quantize oldest
          if len(self.fp16_window) > self.window_size:
              k_old, v_old = self.fp16_window.pop(0)
              # Quantize k_old, v_old to INT4
              self.int4_cache.append((quantize(k_old), quantize(v_old)))
      
      def memory_usage(self):
          fp16_size = len(self.fp16_window) * self.hidden_dim * 2  # bytes
          int4_size = len(self.int4_cache) * self.hidden_dim * 0.5  # bytes
          return fp16_size + int4_size
  
  # Test
  cache = QuantizedKVCache(hidden_dim=768, window_size=512, bit_width=4)
  for i in range(1000):
      k = np.random.randn(768)
      v = np.random.randn(768)
      cache.add_token(k, v)
  
  print(f"Memory: {cache.memory_usage() / 1e6:.1f} MB (vs {1000*768*2*2/1e6:.1f} MB without quantization)")
  ```

**Validation checkpoint**:
- [ ] Calculate: KV cache size with TurboQuant for different window sizes (256, 512, 1024)
- [ ] Explain: Why residual window preserves accuracy (with example)
- [ ] Create: A diagram showing FP16 window + INT4 quantized cache

---

### Step 5: Understand π₀ Quantized Model
**Duration**: 1-2 hours  
**Goal**: Know how π₀ differs from Octo-small and when to use it

**Topics to understand**:
- [ ] π₀ model overview:
  - [ ] Base size: 3B parameters (larger than Octo-small's 500M)
  - [ ] Weight quantization: INT8 (model weights quantized, not just KV cache)
  - [ ] Purpose: Fit in tight memory constraints (even smaller footprint than Octo-small)
- [ ] Double quantization:
  - [ ] Model weights: INT8 (π₀ already quantized)
  - [ ] KV cache: Can apply TurboQuant on top
  - [ ] Combined effect: Very aggressive compression
- [ ] Trade-offs vs Octo-small:
  - [ ] Smaller memory footprint
  - [ ] Slower inference (more quantization overhead)
  - [ ] Potentially lower accuracy (more quantization error)
- [ ] When to use each:
  - [ ] Octo-small: Want balanced accuracy and speed
  - [ ] π₀: Need maximum memory efficiency (e.g., very tight RAM)

**Questions to answer**:
- [ ] How big is π₀ in memory? (vs Octo-small's 2 GB)
  - Answer: ~1-1.5 GB with INT8 weights
- [ ] Why is π₀ called "quantized" if it's larger (3B)?
  - Answer: 3B is raw param count; after INT8 quantization it fits in 1-1.5 GB
- [ ] Should we prioritize Octo-small or π₀ for Pi 4B?
  - Answer: Start with Octo-small (more balanced), validate π₀ as option

**Resources**:
- [ ] Check HuggingFace for π₀ model availability
- [ ] Compare model cards: Octo-small vs π₀
  - Model size (MB)
  - Accuracy benchmarks
  - Inference latency

**Validation checkpoint**:
- [ ] Create a comparison table: Octo-small vs π₀ (size, params, accuracy, speed)
- [ ] Decision: Which model to prioritize for Phase 2?

---

## Learning Checklist

After completing all 5 steps, verify you can answer these:

**Understanding VLAs**:
- [ ] What input does a VLA take? (image + action history)
- [ ] What output does it produce? (next action)
- [ ] Why is this useful for robotics?

**Octo-Small Architecture**:
- [ ] How many parameters total? (500M)
- [ ] What's the hidden dimension? (768)
- [ ] How many transformer layers? (12)
- [ ] How many vision tokens? (~570)

**KV Cache Problem**:
- [ ] What is KV cache? (key and value embeddings from previous tokens)
- [ ] Why cache it? (avoid recomputing attention)
- [ ] How big is Octo-small's KV cache at seq_len=1000? (~37 MB)
- [ ] Why is this a problem on Pi 4B? (limited RAM, causes slowdown/OOM)

**TurboQuant Solution**:
- [ ] What is INT4 quantization? (4 bits per value, 0.25× size)
- [ ] What is a residual window? (recent tokens kept in FP16, old tokens INT4)
- [ ] How much does TurboQuant compress KV cache? (3-4x)
- [ ] Why does it preserve accuracy? (recent tokens FP16, old tokens less important)

**π₀ Model**:
- [ ] How big is π₀? (~1-1.5 GB with INT8 weights)
- [ ] What's different from Octo-small? (weights pre-quantized)
- [ ] When would you use π₀? (need maximum memory efficiency)

---

## Next Steps After Understanding

Once you've completed this learning plan and can answer all checkpoint questions:

1. **Build the mental model**: Create a diagram showing:
   - Octo-small architecture
   - KV cache growth over timesteps
   - TurboQuant compression strategy

2. **Profile the actual models**: 
   - Load Octo-small, measure KV cache size at different seq lengths
   - Estimate memory usage on Pi 4B

3. **Start Phase 1.1**: Read TurboQuant paper with full context
   - You'll understand why each design choice matters
   - Better equipped to implement custom quantizer

4. **Implementation**: Move to Phase 1 implementation with clear understanding

---

## Resources Reference

| Resource | Purpose | Link/Location |
|----------|---------|--------------|
| TurboQuant paper | Algorithm details | arXiv:2504.19874 |
| SGLang PR #21617 | Existing implementation | github.com/sgl-project/sglang |
| LeRobot | Framework for VLA inference | github.com/huggingface/lerobot |
| Octo model card | Architecture details | HuggingFace |
| π₀ model card | Model comparison | HuggingFace |

