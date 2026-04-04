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
