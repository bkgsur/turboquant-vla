# Step 2: Octo-Small Architecture — Reading Materials & Resources

**Duration**: 3-4 hours of focused reading + practical exploration  
**Prerequisites**: Complete Step 1 (VLA Fundamentals)  
**Goal**: Know the exact structure of Octo-small (500M params)

---

## What You'll Learn

By the end of this step, you should know:
- Octo-small's exact architecture (layers, dimensions, parameters)
- What the vision encoder is (ViT-base specifics)
- What the transformer decoder is (12 layers, 768 dims)
- Parameter breakdown (vision: ~86M, decoder: ~400M, action head: <1M)
- How data flows through Octo-small
- Why these specific choices were made

---

## Octo-Small Architecture Overview

### Model Specifications

```
Octo-Small (500M Parameters)

┌─────────────────────────────────────────────┐
│  Vision Encoder: ViT-Base (Frozen)          │
│  - 86M parameters                           │
│  - Input: 256×256 RGB image                 │
│  - Output: 570 visual tokens (768-dim each) │
│  - Training: Pre-trained on ImageNet        │
│  - At inference: FROZEN (no backprop)       │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Transformer Decoder (Trainable)            │
│  - ~400M parameters                         │
│  - 12 layers                                │
│  - 768 hidden dimensions                    │
│  - 12 attention heads (64 dims each)        │
│  - Input: 570 visual + N action tokens      │
│  - Output: Same-sized embeddings            │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Action Head (Linear Layer)                 │
│  - <1M parameters                           │
│  - Input: Processed embeddings              │
│  - Output: Action vector (variable dims)    │
└─────────────────────────────────────────────┘
```

### Parameter Breakdown

```
Total: 500M (500,000,000)

Vision Encoder (ViT-Base):
├─ Patch Embedding: ~2M (converts image patches to embeddings)
├─ Positional Encoding: ~0.5M (position information for patches)
├─ 12 Transformer Blocks: ~83M (each block ~7M)
│  ├─ Multi-head attention (12 heads)
│  ├─ Feed-forward networks
│  └─ Layer normalization
└─ Output projection: ~0.5M

Transformer Decoder (12 layers):
├─ Layer 1-12 (~33M each = 400M total):
│  ├─ Multi-head self-attention: ~12M per layer
│  ├─ Cross-attention (if used): ~12M per layer
│  ├─ Feed-forward: ~9M per layer
│  └─ Layer norm + embeddings
└─ Final layer norm: ~1M

Action Head:
├─ Linear layer (768 → 7): <1M
└─ No activation needed (continuous values)

TOTAL: 86M (vision) + 400M (decoder) + <1M (head) = ~487M
```

---

## Component Deep Dives

### 1. Vision Encoder: ViT-Base

**What is a Vision Transformer (ViT)?**

Traditional approach (CNN):
```
Image (256×256) 
  ↓
Conv layers (extract features hierarchically)
  ↓
Output features
```

Vision Transformer (ViT) approach:
```
Image (256×256)
  ↓
Split into 16×16 patches (256÷16 = 16, so 16×16 = 256 patches)
  ↓
Each patch: 16×16×3 = 768 values → flatten to 768-dim vector
  ↓
Linear projection: 768 → 768 (learned embedding)
  ↓
Add position embeddings (so model knows patch locations)
  ↓
Feed through transformer blocks (same as decoder!)
  ↓
Output: 257 tokens (256 patches + 1 class token for global info)
        Actually 570 tokens in Octo's implementation
```

**Why ViT Instead of CNN?**

| Aspect | CNN | ViT |
|--------|-----|-----|
| **Patch processing** | Hierarchical (lose spatial info) | Uniform (preserve structure) |
| **Parallelization** | Sequential convolutions | Parallel attention |
| **Transfer learning** | ImageNet trained | ImageNet trained (same) |
| **Interpretability** | Hard (black box filters) | Easier (attention weights) |

**ViT-Base Specifics**:
```
- Patch size: 16×16 pixels
- Patches per image: (256÷16) × (256÷16) = 16 × 16 = 256 patches
- Actual in Octo: 570 tokens (may include position tokens, class tokens, etc)
- Hidden dimension: 768
- Attention heads: 12
- Feed-forward dimension: 3072 (4× hidden)
- Layers: 12
- Training data: ImageNet-1K (pre-trained)
- At inference in Octo: Frozen (weights don't change)
```

**Key Insight**: ViT treats image patches like tokens in an LLM. The transformer can attend to any patch from any other patch (not just neighboring pixels like CNN).

---

### 2. Transformer Decoder: 12 Layers

**Architecture of Each Transformer Layer**:

```
Input: 575 tokens (570 visual + 5 action history)
  ↓
┌─────────────────────────────────────┐
│ Layer Normalization                 │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Multi-Head Self-Attention           │
│ - 12 heads                          │
│ - 64 dims per head (768÷12 = 64)    │
│ - Compute attention between all     │
│   575 tokens                        │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Add & Normalize (residual + LN)     │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Feed-Forward Network                │
│ - 768 → 3072 → 768                  │
│ - GELU activation                   │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Add & Normalize (residual + LN)     │
└──────────────┬──────────────────────┘
               ↓
Output: 575 tokens (same size)
```

**Stacked 12 Times** for 12 layers

**Why 12 Layers?**
- More layers → More abstraction (similar to deep CNNs)
- 12 layers is a sweet spot (not too deep, not too shallow)
- ViT-Base uses 12 layers (consistency with encoder)

**Why 768 Dimensions?**
- Standard transformer size (also ViT-Base uses 768)
- Large enough for complex reasoning
- Not so large that inference is slow

---

### 3. Action History Embedding

**How does Octo handle action history?**

```
Previous Actions (e.g., 5 steps):
├─ Action 1: [x1, y1, z1, θ1, grip1] → embed → 768-dim token
├─ Action 2: [x2, y2, z2, θ2, grip2] → embed → 768-dim token
├─ Action 3: [x3, y3, z3, θ3, grip3] → embed → 768-dim token
├─ Action 4: [x4, y4, z4, θ4, grip4] → embed → 768-dim token
└─ Action 5: [x5, y5, z5, θ5, grip5] → embed → 768-dim token

These 5 tokens are concatenated with 570 visual tokens
→ Total input to transformer: 575 tokens
```

**Key Point**: Each action is converted to a 768-dim token, same as visual tokens. This allows the transformer to treat them uniformly.

---

### 4. Action Head: Output Layer

**From Embeddings to Robot Action**:

```
After 12 transformer layers:
- Output: 575 tokens (570 visual + 5 action history)
- Each token: 768-dimensional

Aggregate (e.g., take last action token or average):
- Result: 1 × 768 vector

Linear projection:
- 768 → N (output dimensions, depends on robot)
- W: 768×N matrix (learned)

**For TurboPi specifically**:
- Output: [forward_vel, strafe_vel, rotation, pan_angle, tilt_angle]
- 5 dimensions for mecanum wheels (3) + pan-tilt camera (2)
- forward_vel: [-1, 1] (backward to forward speed)
- strafe_vel: [-1, 1] (left to right strafe)
- rotation: [-1, 1] (counter-clockwise to clockwise)
- pan_angle: [0, 180] (camera pan left-right)
- tilt_angle: [-65, 65] (camera tilt up-down)
```

**Action Output Dimensions**:

Octo is trained on multiple robot platforms, so the output varies:
- **Mobile base robots** (like TurboPi): forward velocity, turning angle, servo angles
- **Arm manipulators**: end-effector position (x, y, z), orientation (roll, pitch, yaw)
- **Variable**: Depends on target robot's action space (mobile platform, arm, grippers, etc.)

The important point: **Action head outputs numbers that directly control a robot**

---

## Data Flow Through Octo-Small

### Example: Single Inference Step

**Scenario**: Robot sees a table with a cup, has lifted it in previous steps

```
Step 1: Capture Image
  Image: 256×256 RGB photo of table with cup being held
  
Step 2: Vision Encoder (ViT-Base)
  Input: 256×256 image
  ├─ Patch embedding: 16×16 = 256 patches + special tokens = 570 tokens
  ├─ 12 transformer blocks process patches
  └─ Output: 570 tokens, 768-dim each
     (compressed representation of the image)

Step 3: Action History Embedding
  Previous 5 actions:
  ├─ Action 1: [0.5, 0.5, 0.5, 0, 0, 0, 0.8] → 768-dim embedding
  ├─ Action 2: [0.5, 0.5, 1.0, 0, 0, 0, 0.8] → 768-dim embedding
  ├─ Action 3: [0.3, 0.5, 1.0, 0, 0, 0, 0.8] → 768-dim embedding
  ├─ Action 4: [0.3, 0.3, 1.0, 0, 0, 0, 0.8] → 768-dim embedding
  └─ Action 5: [0.2, 0.2, 0.9, 0, 0, 0, 0.8] → 768-dim embedding

Step 4: Concatenate
  Combined input: [570 visual tokens] + [5 action history tokens]
                 = 575 total tokens, each 768-dim

Step 5: Transformer Decoder (12 layers)
  ├─ Layer 1: 575 tokens → attend to each other → 575 tokens
  ├─ Layer 2: 575 tokens → attend to each other → 575 tokens
  ├─ ...
  └─ Layer 12: 575 tokens → attend to each other → 575 tokens

Step 6: Action Head
  Input: Last action token (or aggregated output) from layer 12
  Linear projection: 768 → 5 (for TurboPi)
  Output: [forward_vel, strafe_vel, rotation, pan_angle, tilt_angle]
  
Step 7: TurboPi Executes
  Execute predicted action on mecanum wheels + camera servos
  Action: Move forward 0.6 m/s, strafe left slightly, pan camera 45°, tilt 20°
```

---

## Memory & Compute Requirements

### Memory Footprint

```
Model Weights (FP16):
├─ Vision Encoder (ViT): 86M params × 2 bytes = 172 MB
├─ Transformer Decoder: 400M params × 2 bytes = 800 MB
├─ Action Head: <1M params × 2 bytes = <2 MB
└─ Total: ~974 MB ≈ 1 GB (FP16)

Or: ~2 GB (FP32), ~500 MB (INT8)

KV Cache (during inference, THIS IS THE PROBLEM):
├─ Sequence length: 575 tokens (without action history: 570)
├─ With action history growing: 570 + N actions
├─ Per layer: 2 × seq_len × hidden_dim × 2 bytes (K and V, FP16)
├─ For layer: 2 × 575 × 768 × 2 = 1.77 MB
├─ For 12 layers: 1.77 MB × 12 = 21.2 MB
│
└─ But sequence grows! With 1000 action history tokens:
   ├─ Per layer: 2 × 1570 × 768 × 2 = 4.8 MB
   ├─ For 12 layers: 4.8 MB × 12 = 57.6 MB
   └─ Problem: KV cache keeps growing!

THIS IS WHY TURBOQUANT IS NEEDED:
═══════════════════════════════════════
Without compression:
└─ Model (2GB) + Activations (400MB) + KV cache (50+MB) 
   = 2.5+ GB on Pi 4B → TIGHT!

With TurboQuant (4-bit KV compression):
└─ Model (2GB) + Activations (400MB) + KV cache (12-15MB)
   = 2.4 GB on Pi 4B → FITS WITH MARGIN!
```

### Inference Latency on RTX 4050

**Rough estimates** (actual depends on implementation):
```
Vision Encoder (ViT-Base):
├─ 570 patches → 12 transformer layers
└─ ~50-100 ms

Transformer Decoder (12 layers × 575 tokens):
├─ Self-attention: ~75-150 ms
├─ Feed-forward: ~25-50 ms
└─ Total: ~100-200 ms

Action Head:
└─ ~1-5 ms

Total per inference: ~150-300 ms (3-6 Hz on GPU)

On Pi 4B (ARM Cortex-A72):
└─ Estimated: 300-500 ms (~2-3 Hz) without optimization
   
With TurboQuant:
└─ Estimated: 200-300 ms (3-5 Hz) due to smaller KV cache
```

---

## Practical Understanding Exercises

### Exercise 1: Count Parameters

**Task**: Given Octo-small's 500M parameters, distribute them among:
- Vision Encoder
- Transformer Decoder
- Action Head

**Expected breakdown**:
```
Vision Encoder: 86M (17%)
Transformer Decoder: 400M (80%)
Action Head: <1M (0.2%)
Other embeddings: ~13M (2.6%)
Total: ~500M ✓
```

### Exercise 2: Trace Token Flow

**Task**: If you input:
- 1 image (→ 570 tokens)
- Action history of 10 steps (→ 10 tokens)

How many tokens enter the transformer?

**Answer**: 570 + 10 = 580 tokens

**Follow-up**: If action history grows to 100 steps?

**Answer**: 570 + 100 = 670 tokens  
**Impact**: KV cache grows to ~2.6 MB per layer (vs 2.1 MB for 580 tokens)

### Exercise 3: Memory Calculation

**Task**: Calculate KV cache size for Octo-small:
- 12 layers
- Hidden dim: 768
- Sequence length: 1000 tokens
- Data type: FP16 (2 bytes per value)

**Formula**: 2 × seq_len × hidden_dim × 2 bytes × num_layers

**Calculation**:
```
2 × 1000 × 768 × 2 × 12
= 2 × 1000 × 768 × 24
= 36,864,000 bytes
= 36.8 MB per layer
= 36.8 × 12 = ??? MB total (WRONG - don't multiply by 12 again!)

Correct:
= 2 × 1000 × 768 × 2 × 12 = 36,864,000 bytes ÷ 1,000,000 = 36.8 MB

For 12 layers: Need to multiply per-layer calculation by 12
= (2 × 1000 × 768 × 2) × 12 = 3,072,000 × 12 = 36,864,000 bytes = 36.8 MB

Actually: 2 (K and V) × seq_len × hidden_dim × 2 bytes × num_layers
= 2 × 1000 × 768 × 2 × 12 = 36.8 MB ✓
```

**With TurboQuant (3-4x compression)**:
```
36.8 MB ÷ 4 = 9.2 MB
```

---

## Resources

### Official Documentation

1. **Octo Model Card** (if available on HuggingFace):
   - Architecture details
   - Training data
   - Performance benchmarks
   - https://huggingface.co/google/octo-base-1B

2. **Vision Transformer (ViT) Paper**:
   - Title: "An Image is Worth 16x16 Words"
   - arXiv: https://arxiv.org/abs/2010.11929
   - Read: Abstract + Section 3.1 (Vision Transformer Architecture)
   - Time: 20-30 mins

3. **Transformer Architecture Paper**:
   - Title: "Attention Is All You Need"
   - arXiv: https://arxiv.org/abs/1706.03762
   - Read: Section 3 (Architecture), Section 3.2 (Multi-Head Attention)
   - Time: 30-40 mins

### Code Resources

1. **LeRobot Octo Implementation**:
   - GitHub: https://github.com/huggingface/lerobot
   - File: `lerobot/policies/transformers/octo/` (if available)
   - Look for: Model definition, forward pass
   - Time: 30-45 mins of code reading

2. **PyTorch Transformer Implementation**:
   - Official: `torch.nn.Transformer`
   - Good for understanding the decoder loop

3. **Simple Transformer Implementation**:
   - Blog: https://towardsdatascience.com/attention-is-all-you-need-8751d3f3e3f8
   - Great visual explanations
   - Time: 30-45 mins

---

## Validation Checklist

Can you answer these?

- [ ] **Q1**: How many parameters in Octo-small?  
  **A**: 500M (500 million)

- [ ] **Q2**: What's the vision encoder in Octo?  
  **A**: ViT-Base (pre-trained on ImageNet, frozen at inference)

- [ ] **Q3**: How many transformer layers in the decoder?  
  **A**: 12 layers

- [ ] **Q4**: What's the hidden dimension?  
  **A**: 768

- [ ] **Q5**: How many visual tokens from ViT?  
  **A**: ~570 tokens (each 768-dim)

- [ ] **Q6**: How many transformer heads?  
  **A**: 12 heads (64 dims each: 768÷12=64)

- [ ] **Q7**: How many output dimensions for robot action?  
  **A**: Variable (depends on robot platform — TurboPi uses mobile base + servo angles, arms use xyz + orientation, etc.)

- [ ] **Q8**: If action history is 50 steps, how many tokens enter transformer?  
  **A**: 570 (visual) + 50 (history) = 620 tokens

- [ ] **Q9**: What's the KV cache size at seq_len=1000?  
  **A**: ~36.8 MB (for FP16, 12 layers)

- [ ] **Q10**: Why is vision encoder frozen?  
  **A**: Reuse ImageNet pre-training; focus training on decoder for action prediction

---

## Summary

**Octo-Small Key Facts**:
- 500M parameters total
- Vision Encoder: ViT-Base (86M, frozen)
- Transformer Decoder: 12 layers, 768 hidden dim (400M)
- Action Head: Variable output dimensions for different robots (linear layer, <1M)
- Inference: 150-300ms on GPU, 300-500ms on Pi
- KV Cache: Grows with action history length (THIS IS THE PROBLEM!)

**Why These Choices**:
- ViT-Base: Pre-trained vision knowledge from ImageNet
- 12 layers: Proven depth for vision transformers
- 768 dims: Good balance of expressiveness and efficiency
- Variable output: Supports many robot platforms (arms, mobile bases, TurboPi, etc.)

**Next Step**: Step 3 — KV Cache Growth Problem

---

**Time spent on Step 2**: ~3-4 hours  
**Confidence check**: Can you draw Octo-small's architecture from memory?  
**Next**: Step 3 — KV Cache Growth (Why Memory Explodes)
