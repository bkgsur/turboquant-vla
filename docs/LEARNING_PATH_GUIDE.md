# Learning Path Guide — Start Here!

This guide directs you through the 5-step learning plan to understand Octo-small, π₀, and TurboQuant KV cache compression.

**Total Time**: ~11-20 hours (2-4 hours per step)

---

## 📚 Available Reading Materials

### ✅ Step 1: VLA Fundamentals (2-3 hours)
**File**: [step1_vla_fundamentals.md](step1_vla_fundamentals.md)  
**Status**: ✅ READY NOW

**Learn**: What are VLAs? How are they different from LLMs? Why action history matters?

**Key Concepts**:
- Vision-Language-Action models
- Vision Encoder → Transformer → Action Head pipeline
- Action history and temporal context
- Bounded vs unbounded sequences

**Reading checklist**:
- [ ] Core Reading Material section (30 mins)
- [ ] Key Concepts Checklist (10 mins)
- [ ] Practical Understanding Exercises (45 mins)
- [ ] Resources + External reading (60-90 mins)

**Validation**: Can you explain "VLAs predict the next robot action based on image and history"?

---

### ✅ Step 2: Octo-Small Architecture (3-4 hours)
**File**: [step2_octo_architecture.md](step2_octo_architecture.md)  
**Status**: ✅ READY NOW

**Learn**: Exact structure of Octo-small (500M params, 12 layers, 768 hidden_dim)

**Key Concepts**:
- Vision Encoder: ViT-Base (570 tokens, 86M params)
- Transformer Decoder: 12 layers, 768 hidden dims
- Action Head: 7D output
- Parameter breakdown and memory requirements
- KV cache growth problem (preview)

**Reading checklist**:
- [ ] Architecture Overview section (30 mins)
- [ ] Component Deep Dives (90 mins)
- [ ] Data Flow Example (30 mins)
- [ ] Memory & Compute section (30 mins)
- [ ] Practical Exercises (30 mins)
- [ ] Code exploration (optional, 30-45 mins)

**Validation**: Can you list Octo-small's exact specs (params, layers, hidden_dim, output)?

---

### ⏳ Step 3: KV Cache Growth Problem (2-3 hours)
**File**: [step3_kv_cache_problem.md](step3_kv_cache_problem.md)  
**Status**: COMING SOON (will create after you complete Steps 1-2)

**Learn**: Why KV cache explodes during inference, memory calculations, why it's a problem

---

### ⏳ Step 4: TurboQuant Solution (3-4 hours)
**File**: [step4_turboquant_solution.md](step4_turboquant_solution.md)  
**Status**: COMING SOON (will create after you complete Steps 1-3)

**Learn**: How TurboQuant solves KV cache problem, quantization mechanics, residual window

---

### ⏳ Step 5: π₀ Quantized Model (1-2 hours)
**File**: [step5_pi0_model.md](step5_pi0_model.md)  
**Status**: COMING SOON (will create after you complete Steps 1-4)

**Learn**: π₀ model structure, double quantization, when to use each model

---

## 🎯 How to Use This Guide

### Start Here (Right Now!)

1. **Read this file completely** (5 mins) ← You are here
2. **Read Step 1: VLA Fundamentals** (2-3 hours)
   - Open: [step1_vla_fundamentals.md](step1_vla_fundamentals.md)
   - Take your time, don't rush
   - Do the exercises
   - Answer validation questions

3. **After Step 1, read Step 2: Octo Architecture** (3-4 hours)
   - Open: [step2_octo_architecture.md](step2_octo_architecture.md)
   - Focus on understanding parameter counts and token flow
   - Do the exercises and memory calculations

4. **Tell me when you've completed Steps 1-2**, and I'll create Steps 3-4-5

### Reading Tips

1. **Take notes**: Keep a notebook or text file for key concepts
2. **Do the exercises**: Don't just read, actively work through examples
3. **Answer validation questions**: Check your understanding at the end
4. **Look at external resources**: Links are provided, use them
5. **Don't memorize**: Understand the concepts, you can look up details later

---

## 📖 Reference Materials Also Available

These provide context and support your learning:

1. **[docs/pi4b_specs.md](pi4b_specs.md)**
   - Raspberry Pi 4B hardware specs (ARM Cortex-A72, 4GB RAM)
   - Constraints you need to understand
   - **Read when**: Before starting Phase 1 implementation

2. **[docs/understanding_plan.md](understanding_plan.md)**
   - Original 5-step learning outline
   - **Read when**: Quick reference of learning goals

3. **[docs/model_analysis.md](../docs/understanding_plan.md)**
   - Deep analysis of models and KV cache
   - **Read when**: Want deeper technical understanding

4. **[CLAUDE.md](../CLAUDE.md)**
   - Project guidance and collaboration guidelines
   - **Read when**: Before working with Claude Code

---

## 🔬 Foundation Papers Summary

Three seminal papers provide the theoretical foundation for TurboQuant VLA. Below is a concise summary of each and how it relates to your learning path.

### **Paper 1: "Attention Is All You Need" (Vaswani et al., 2017)**

**Why it matters**: Introduces the Transformer architecture—the foundation of all modern VLAs including Octo.

**Key concepts for TurboQuant**:
- **Self-Attention Mechanism**: How transformers attend to different parts of input
- **KV Cache in Inference**: During autoregressive decoding, Key-Value caches grow **linearly with sequence length** → this is your primary optimization target
- **Multi-Head Attention**: 8-16 attention heads per layer; quantization strategy may differ per head
- **Complexity**: O(n²·d) per layer where n = sequence length, d = hidden dimension

**When to read**: 
- Step 1: Read abstract + Section 3 (20 mins) for mechanism understanding
- Step 3: Return for KV cache mechanics when learning about the problem

---

### **Paper 2: "An Image is Worth 16×16 Words" (Dosovitskiy et al., 2021)**

**Why it matters**: Vision Transformer (ViT) is the backbone for vision encoding in Octo and other VLAs.

**Key concepts for TurboQuant**:
- **Patch Tokenization**: Images split into 16×16 patches → 196-576 tokens per image (vs millions of pixels)
- **Scalability**: Requires large pretraining datasets (14M-300M images) to outperform CNNs
- **Architecture Details**: 12 layers, 768 hidden_dim, learned position embeddings
- **Transfer Learning**: Pretrain on large data → fine-tune to downstream tasks with minimal data

**Relevance to your project**:
- Octo-small inherits ViT's vision encoding (image patches + language tokens)
- Sequence grows with: **history frames (2×570 tokens) + language tokens (16) + action tokens = 1156+ tokens per step**
- Over 10-20 steps of history → KV cache can exceed 4GB on Pi 4B!

**When to read**: 
- Step 1: Read abstract + Section 3.1 (15 mins) for vision encoding
- Step 2: Deep dive into architecture when studying Octo's exact structure

---

### **Paper 3: "Octo: An Open-Source Generalist Robot Policy" (Octo Model Team, 2024)**

**Why it matters**: This IS your target model. Octo-small (500M params) is what you'll optimize with TurboQuant.

**Key architecture**:
- **Input Tokenization**:
  - Vision: 3rd-person camera (256 tokens) + wrist camera (256 tokens) + language instructions (16 tokens)
  - Proprioception: Joint positions, velocities (8 tokens)
  - Total: 2100+ tokens per inference step
- **Transformer Backbone**: 12-24 layers, ViT-style block-wise masked attention
- **Action Head**: Diffusion policy for continuous action prediction
- **Finetuning**: Adapts to new robots in ~5 hours on A5000 GPU with 100 demonstrations

**KV Cache Explosion**:
- Each attention layer stores K,V: (seq_len × hidden_dim) per head
- With 2100 tokens × 12 layers × 12 heads → **massive memory footprint**
- Octo-small: 27M-93M params; Runtime KV cache can be **2-3 GB** with history
- **Your job**: Compress this to <1 GB using quantization

**When to read**: 
- Step 2: Focus on architecture (Section III.A) and design decisions
- Step 3: Return for KV cache growth calculation

---

## 📊 How Papers Connect to Learning Steps

```
Step 1: VLA Fundamentals
  └─ Read Attention paper abstract (why transformers work)

Step 2: Octo-Small Architecture
  └─ Read ViT paper Section 3 (vision encoding)
  └─ Read Octo paper Section III (exact architecture)

Step 3: KV Cache Problem
  └─ Return to Attention paper for KV mechanics
  └─ Calculate memory using Octo's architecture specs

Step 4: TurboQuant Solution
  └─ Understand residual window: keep recent tokens (FP16),
     quantize old tokens (INT4)

Step 5: π₀ Model
  └─ Compare Octo-small vs π₀ quantized (different param counts)
```

---

## 🎯 Key Insight: Why TurboQuant Matters

| What | Problem | TurboQuant Solution |
|------|---------|---------------------|
| **KV Cache Growth** | Grows 2×seq_len×hidden_dim per layer | Keep recent 256 tokens FP16, quantize older tokens INT4 |
| **Memory on Pi 4B** | Octo needs 2-3 GB just for cache | Target: <1 GB through 3-4× compression |
| **Inference Speed** | Attention is O(n²) complexity | Quantization reduces cache size → faster operations |
| **Accuracy** | Quantization loses precision | Residual window keeps important recent context |

**The chain**: Transformers (paper 1) → ViT (paper 2) → Octo (paper 3) → **TurboQuant** (your project)

---

## 📖 When to Read Papers During This Learning Plan

- **Right now (Step 1)**: Read paper abstracts (10 mins total) to understand the motivation
- **Step 2**: Deep dive into Octo paper Section III (30 mins)
- **Step 3**: Read Attention paper for KV cache mechanics (40 mins)
- **Step 4-5**: Return to papers for reference as needed

**Note**: You don't need to read the full papers; focus on sections relevant to KV cache and architecture. Full papers are provided in `/docs/` for reference.

---

## ✅ Progress Checklist

Track your progress here:

### Step 1: VLA Fundamentals
- [ ] Read core material
- [ ] Review key concepts
- [ ] Complete exercises
- [ ] Answer validation questions
- **Status**: Not started
- **Completion Date**: ___________

### Step 2: Octo-Small Architecture  
- [ ] Read architecture overview
- [ ] Understand component deep dives
- [ ] Trace data flow example
- [ ] Do memory calculations
- [ ] Complete exercises
- **Status**: Not started
- **Completion Date**: ___________

### Step 3: KV Cache Growth Problem
- [ ] (Will be created soon)
- **Status**: Waiting for Steps 1-2
- **Completion Date**: ___________

### Step 4: TurboQuant Solution
- [ ] (Will be created soon)
- **Status**: Waiting for Steps 1-3
- **Completion Date**: ___________

### Step 5: π₀ Model
- [ ] (Will be created soon)
- **Status**: Waiting for Steps 1-4
- **Completion Date**: ___________

---

## 📞 Getting Help

**If you're stuck on a concept**:

1. **Re-read the relevant section** — Understanding takes time
2. **Check the exercises** — Often clarify concepts
3. **Look at external resources** — Links provided in each step
4. **Ask me questions** — I can clarify confusing parts
5. **Take a break** — Mental breaks improve learning

**Common "stuck" points**:
- **Attention mechanism** → Watch 3Blue1Brown or StatQuest video (15-20 mins)
- **Transformers** → Read "Attention Is All You Need" Section 3 (30 mins)
- **ViT** → Read ViT paper abstract + section 3.1 (20 mins)
- **Memory math** → Work through examples slowly, use calculator

---

## 🎓 Learning Outcomes

**After completing all 5 steps, you'll be able to**:

✅ **Explain a VLA to someone in 2 minutes**
- What it sees, what it remembers, what it does

✅ **List Octo-small's exact architecture**
- 500M params, ViT-Base encoder, 12-layer decoder, 7D output

✅ **Calculate KV cache size at any sequence length**
- Memory formula: 2 × seq_len × hidden_dim × 2 × num_layers

✅ **Describe how TurboQuant works**
- Residual window (FP16 recent) + quantized history (INT4)

✅ **Understand why TurboQuant matters for Pi 4B**
- 4GB RAM constraint → KV cache compression is essential

✅ **Know when to use Octo-small vs π₀**
- Trade-offs: accuracy vs memory vs speed

---

## 🚀 After Learning Plan

Once you complete all 5 steps:

1. **You're ready for Phase 1 implementation**
   - See: [implementation_plan.md](../implementation_plan.md)
   - Task 1.1: Study TurboQuant paper (understand algorithm)
   - Task 1.2: Analyze existing implementations
   - Task 1.3: Build standalone TurboQuant library

2. **You can read research papers** on the topic
   - TurboQuant paper: arXiv:2504.19874
   - You'll understand the concepts without getting lost

3. **You can read code** and understand what's happening
   - LeRobot implementations
   - Transformer libraries
   - Attention mechanisms

---

## 📝 Recommended Study Schedule

**Option A: Intensive (1 week)**
- Day 1: Step 1 (2-3 hours)
- Day 2: Step 2 (3-4 hours)
- Day 3: Step 3 (2-3 hours)
- Day 4: Step 4 (3-4 hours)
- Day 5: Step 5 (1-2 hours)
- Days 6-7: Review + Phase 1 preparation

**Option B: Moderate (2 weeks)**
- Week 1: Steps 1-2 (5-7 hours)
  - Day 1-2: Step 1
  - Day 3-4: Step 2
  - Day 5: Review and exercises
- Week 2: Steps 3-5 (7-9 hours)
  - Day 1-2: Step 3
  - Day 3-4: Step 4
  - Day 5-6: Step 5
  - Day 7: Review + Phase 1 prep

**Option C: Relaxed (3-4 weeks)**
- Learn at your own pace
- Do all exercises thoroughly
- Read external resources deeply
- Build strong foundation for implementation

---

## 🎯 Quick Start (Right Now!)

**Do this:**

1. ✅ You're reading this file
2. ⏭️ Next: Open [step1_vla_fundamentals.md](step1_vla_fundamentals.md)
3. 📖 Read "Core Reading Material" section (30 mins)
4. 📝 Do the exercises (45 mins)
5. ✔️ Answer validation questions
6. 📢 Tell me when done → I'll create Step 3!

---

## Summary

| Step | Topic | Duration | Status |
|------|-------|----------|--------|
| 1 | VLA Fundamentals | 2-3 hrs | ✅ Ready |
| 2 | Octo Architecture | 3-4 hrs | ✅ Ready |
| 3 | KV Cache Problem | 2-3 hrs | ⏳ Coming |
| 4 | TurboQuant Solution | 3-4 hrs | ⏳ Coming |
| 5 | π₀ Model | 1-2 hrs | ⏳ Coming |
| **Total** | | **11-20 hrs** | **5 ready, 3 coming** |

---

**Ready to start?** 👇

## 👉 Next: [Open Step 1: VLA Fundamentals](step1_vla_fundamentals.md)

Start with the "Core Reading Material" section. Take your time, do the exercises, and enjoy learning!

Once you've completed Step 1, move to Step 2. After both are done, let me know and I'll create the remaining steps.

---

**Questions?** Feel free to ask me to clarify any concepts as you read!

**Good luck! 🚀**
