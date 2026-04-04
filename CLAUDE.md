# CLAUDE.md — TurboQuant VLA Project Guidance

## Project Overview

**TurboQuant for Raspberry Pi 4B with TurboPi**

Deploy TurboQuant KV cache compression to enable real-time VLA inference on HiWonder TurboPi.

**Hardware Specification**:
- **Device**: HiWonder TurboPi (Robotic platform)
- **SBC**: Raspberry Pi 4B
- **Processor**: Quad-core ARM Cortex-A72 @ 1.5 GHz
- **Memory**: 4GB RAM
- **Target**: 5-10 Hz inference latency with <5% accuracy loss

**Target models**: 
- Primary: Octo-small (500M params)
- Backup: π₀ quantized (3B params, pre-quantized)

**Duration**: 5 phases (no time estimates)  
**Primary deliverable**: Production-ready TurboQuant KV cache library optimized for ARM Cortex-A72 + LeRobot integration

---

## Core Context

### Problem Statement
- VLA models need 24GB+ VRAM due to KV cache growth during autoregressive decoding
- KV cache explodes with action history (100-2000+ tokens)
- Raspberry Pi 4B has only 4GB RAM → models don't fit or run at <1 Hz
- **Solution**: TurboQuant reduces KV cache 3-4x via residual window quantization (recent tokens FP16, old tokens INT4)

### Why This Matters
- **Robotics bottleneck**: Real-time control needs 5-10 Hz minimum (100-200ms per action)
- **Cost**: Pi 4B (~$75) vs Jetson AGX Orin (~$800) — 10x cheaper
- **Deployment**: Portable, low-power solution for edge robotics on resource-constrained hardware

### Model Landscape
- **Octo-small**: 500M params, 768 hidden_dim, 12 layers. Baseline choice (balanced accuracy/speed).
- **π₀ quantized**: 3B raw params, INT8 weights (~1-1.5 GB). For extreme memory constraints.
- **Both**: Benefit from TurboQuant KV cache compression (additional 3-4x savings).

---

## Documentation Structure

- **[plan.md](plan.md)** — High-level goals, hardware, tech stack, research questions
- **[implementation_plan.md](implementation_plan.md)** — Detailed 5-phase roadmap with tasks, deliverables, success criteria
- **[docs/understanding_plan.md](docs/understanding_plan.md)** — Learning path to understand models + KV cache compression (prerequisite reading)

---

## Development Guidelines

### Before Starting Phase 1
1. **Complete the learning plan** in `docs/understanding_plan.md` (5 steps)
   - Understand VLAs, Octo-small architecture, KV cache growth, TurboQuant mechanics, π₀ model
   - Don't skip this — it ensures you understand *why* we're building what we're building
2. **Create reference artifacts**:
   - Architecture diagram (Octo-small with layer counts)
   - KV cache growth graph (memory vs sequence length, with/without TurboQuant)
   - Quantization explanation (residual window strategy)
3. **Verify model availability**: Check HuggingFace for Octo-small and π₀ models before implementation

### Code Standards

**Structure**: Follow the planned directory layout in `implementation_plan.md` (File Structure section):
```
turboquant_kv/          # Core library
├── quantizer.py        # KVQuantizer class
├── calibration.py      # Calibrator class
├── kernels/            # Optimized kernels
├── config.py           # QuantizerConfig dataclass
└── tests/              # Unit tests (>90% coverage)

octo_integration/       # Octo-small integration
lerobot_integration/    # LeRobot integration
scripts/                # Benchmarks, profiling scripts
docs/                   # Design docs, benchmark results
```

**Conventions**:
- **Type hints**: All functions must have type annotations
- **Docstrings**: All public classes/functions (Google style)
- **Tests**: Unit tests for quantization round-trip, accuracy, memory
- **Benchmarks**: Track memory reduction (target: 4x for KV cache), latency, accuracy
- **No code without docs**: Every module needs a docstring explaining its purpose

### Testing Strategy

1. **Unit tests** (`tests/`):
   - Quantization round-trip: quantize → dequantize → close to original
   - Memory savings: measure actual compression ratio
   - Accuracy: action prediction MSE on test set

2. **Integration tests** (`scripts/`):
   - Octo-small + TurboQuant on RTX 4050 (baseline)
   - Octo-small + TurboQuant on actual Pi 4B (real hardware)
   - End-to-end inference with LeRobot

3. **Success criteria** (from implementation_plan.md):
   - Phase 1: 4x KV cache reduction, <5% accuracy loss
   - Phase 2: 5-10 Hz on Pi 4B, <3.5 GB memory
   - Phase 4: Full pipeline on TurboPi, task execution validated

### Benchmarking

**Always measure**:
1. **Memory**: KV cache size (bytes) at different sequence lengths
2. **Latency**: Inference time per action (ms), target: 100-200ms for 5-10 Hz
3. **Accuracy**: Action prediction MSE or task success rate

**Document in**:
- `docs/pi_benchmark_results.md` (Phase 2)
- `docs/pi_profiling_report.md` (Phase 3)
- `docs/turbopi_test_results.md` (Phase 4)

---

## Key Dependencies & Constraints

### Required Knowledge
- **Transformers**: Attention mechanism, multi-head self-attention, KV cache mechanics
- **Quantization**: INT4/INT3 encoding, dequantization, calibration
- **PyTorch**: Tensor operations, autograd (if adding trainable quantization)
- **ARM deployment**: ONNX Runtime, TorchScript, ARM SIMD considerations

### Hardware Constraints (Raspberry Pi 4B — THE TARGET)
- **Processor**: Quad-core ARM Cortex-A72 @ 1.5 GHz (no GPU acceleration)
- **Memory**: 4GB RAM (total system, not just for inference)
- **Thermal**: Active heatsink recommended (throttles at >80°C)
- **Power**: ~3-5W typical, up to 15W under load

**Development Machine** (RTX 4050 for simulation only):
- Used to simulate Pi's memory constraints
- NOT the deployment target
- Helps prototype before Pi testing

### Software Constraints
- **Python 3.9+** (uv environment)
- **PyTorch**: CPU-capable (Pi), GPU-capable (RTX 4050)
- **No external quantization frameworks**: Build custom quantizer (vs relying on external pkg)
- **ONNX export**: For ARM deployment (Phase 3)

---

## Implementation Order (Don't Skip Phases)

1. **Phase 1: Core Library** — Build TurboQuant quantizer in isolation
   - Don't integrate with models yet
   - Focus on quantization mechanics, unit tests, memory savings
2. **Phase 2: Octo Integration** — Apply to Octo-small, validate on Pi 4B
   - Measure latency, memory, accuracy
   - This is the MVP (minimum viable product)
3. **Phase 3: ONNX & Optimization** — Export to ONNX, ARM optimization
   - Profile on Pi, identify bottlenecks
4. **Phase 4: LeRobot Integration** — Real robot deployment
   - End-to-end testing on TurboPi
5. **Phase 5: Documentation** — Polish, deploy guide, reproducible setup

**Skipping phases = risks**:
- Skip Phase 1 → Quantizer incomplete, breaks Phase 2
- Skip Phase 2 → No validation on real hardware, late discovery of issues
- Skip Phase 3 → Inference too slow on Pi, undeployable
- Skip Phase 4 → Can't verify real robot works

---

## Collaboration & Feedback

### When You're Stuck
1. **Quantization accuracy dropping**: Check residual window size, calibration data quality
2. **Memory not improving**: Verify quantization is actually applied (not just computed)
3. **Inference slow on Pi**: Profile first (don't guess), then optimize bottleneck
4. **Model unavailable**: Fall back to similar-sized alternatives (search HuggingFace)

### Decision Points (Ask User)
- **Residual window size**: 256 vs 512 vs 1024? (affects accuracy/speed)
- **Bit-width**: 3-bit vs 4-bit? (affects memory/accuracy)
- **Calibration data**: How much? 50 vs 100 vs 200 examples?
- **Model priority**: Octo-small or π₀ first?

---

## Success Metrics (End of Project)

✅ **Phase 1 Complete**: TurboQuant library standalone, 4x KV cache reduction, >90% test coverage  
✅ **Phase 2 Complete**: Octo-small + TurboQuant runs on Pi 4B at 5-10 Hz, <3.5 GB memory, <5% accuracy loss  
✅ **Phase 3 Complete**: ONNX model optimized for ARM, profiling report shows bottlenecks  
✅ **Phase 4 Complete**: Full pipeline on TurboPi, task execution validated  
✅ **Phase 5 Complete**: Documented, reproducible deployment, polished code  

**Extra credit**: π₀ model support, ROS 2 integration, weight quantization

---

## Related Resources

**This Project**:
- **[docs/pi4b_specs.md](docs/pi4b_specs.md)** — Raspberry Pi 4B constraints & performance (CRITICAL READING)
- **[docs/understanding_plan.md](docs/understanding_plan.md)** — Learning path for models and quantization

**External**:
- **TurboQuant paper**: arXiv:2504.19874
- **SGLang implementation**: github.com/sgl-project/sglang/pull/21617
- **LeRobot framework**: github.com/huggingface/lerobot
- **OpenVLA models**: github.com/openvla/openvla
- **Octo model**: HuggingFace (huggingface.co/google/octo-base-1B or similar)
- **π₀ model**: HuggingFace (search for pi-zero or quantized VLA)

---

## Project Assumptions (Confirm Before Building)

1. ✅ Octo-small model exists and can be loaded from HuggingFace
2. ⚠️ TurboQuant quantization is architecture-agnostic (mostly true, but verify)
3. ⚠️ Residual window of 512 tokens sufficient (may need tuning)
4. ⚠️ INT4 precision adequate for robotics tasks (test early!)
5. ⚠️ ONNX Runtime available for ARM (check Pi support)

---

## File Naming & Organization

**Code**:
- `turboquant_kv/` — Package (namespace for distribution)
- `octo_integration/` — Integration module
- `lerobot_integration/` — Integration module
- `scripts/` — Standalone scripts (benchmarks, profiling, demos)

**Documentation**:
- `docs/` — Design docs, results, guides
- `examples/` — Jupyter notebooks, working examples
- `README.md` — Quick start, installation
- `CLAUDE.md` — This file (project guidance)

**Data & Config**:
- `.env.template` — Environment variables template (rename to `.env` locally)
- `pyproject.toml` — Project metadata, dependencies, uv config

---

## Questions to Answer at Each Phase Boundary

**After Phase 1**: 
- Can we quantize and dequantize KV cache without loss of accuracy?
- Do we achieve 4x compression on RTX 4050?

**After Phase 2**: 
- Does Octo-small + TurboQuant fit on Pi 4B in 4GB RAM?
- Is inference speed 5-10 Hz (achievable)?
- Is accuracy loss < 5%?

**After Phase 3**: 
- What's the latency bottleneck on Pi? (matrix multiply? softmax? dequantization?)
- Can ONNX Runtime improve latency?

**After Phase 4**: 
- Does the full pipeline work on real TurboPi hardware?
- Can the robot execute tasks (e.g., pick & place) with quantized model?

---

## Ground Rules

✅ **DO**:
- Read the learning plan before implementing
- Benchmark early and often
- Test on actual Pi hardware (don't just simulate)
- Document design decisions
- Keep success criteria in mind

❌ **DON'T**:
- Skip phases or validation steps
- Guess at performance — profile first
- Use time estimates (they're unreliable)
- Over-engineer (keep it simple until you need complexity)
- Deploy without testing on real hardware

---

**Last Updated**: 2026-04-04  
**Project Owner**: Suresh Gopalakrishnan  
**Status**: Planning → Phase 1 (Learning Plan prerequisite)
