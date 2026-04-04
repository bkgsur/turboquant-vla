# Documentation

Design documents, architecture notes, benchmarking results, and deployment guides.

## Core Documentation

- **`understanding_plan.md`** — Learning roadmap to understand models, KV cache, and TurboQuant
  - 5 steps: VLA basics, Octo-small architecture, KV cache growth, TurboQuant mechanics, π₀ model
  - Prerequisite reading before Phase 1 implementation

- **`turboquant_architecture.md`** — (Phase 1 deliverable) TurboQuant algorithm deep dive
  - Quantization mechanics (per-channel, per-token, symmetric/asymmetric)
  - Residual window strategy
  - Calibration procedure
  - Expected memory savings for Octo-small and π₀

- **`implementation_options.md`** — (Phase 1 deliverable) Comparison of quantization approaches
  - SGLang PR #21617 (existing implementation)
  - turboquant PyPI package (existing implementation)
  - Custom implementation (build from scratch)
  - Decision matrix and reasoning

## Phase-Specific Results

### Phase 1: Core Library
- *(Pending)* `turboquant_architecture.md` — Algorithm design
- *(Pending)* `implementation_options.md` — Comparison of approaches

### Phase 2: Model Integration & Pi Validation
- *(Pending)* `octo_benchmark_baseline.md` — Octo-small FP16 baseline on RTX 4050
- *(Pending)* `pi_benchmark_results.md` — Octo-small + TurboQuant on Pi 4B
  - Inference latency (per-action time, throughput)
  - Memory usage (peak, timeline)
  - Thermal behavior (CPU temp, throttling)
  - Accuracy (action MSE, task success rate)

- *(Pending)* `accuracy_validation.md` — Accuracy impact of quantization
  - Action prediction MSE (quantized vs baseline)
  - Bit-width comparison (3-bit vs 4-bit)
  - Residual window size sensitivity

### Phase 3: ONNX Export & Optimization
- *(Pending)* `pi_profiling_report.md` — Detailed profiling on Pi
  - Layer-by-layer latency breakdown
  - Memory allocation timeline
  - Bottleneck identification (matrix multiply? softmax? dequantization?)
  - ONNX vs PyTorch latency comparison

- *(Pending)* `arm_optimization_guide.md` — Optimizations for ARM Cortex-A72
  - ONNX Runtime configuration
  - SIMD considerations
  - Weight quantization trade-offs

### Phase 4: Robot Deployment
- *(Pending)* `turbopi_test_results.md` — End-to-end testing on TurboPi
  - Inference latency, memory, thermal
  - Task execution results (if applicable)
  - Comparison: quantized vs baseline (if baseline fits)
  - Integration issues and solutions

- *(Pending)* `turbopi_deployment.md` — Deployment guide
  - Setup instructions for Pi 4B
  - Package installation
  - Model loading and inference
  - Troubleshooting common issues

### Phase 5: Documentation
- *(Pending)* `performance_guide.md` — Tuning guide for custom models
  - How to apply TurboQuant to new models
  - Choosing bit-width and window size
  - Calibration best practices
  - Expected accuracy loss

- *(Pending)* `troubleshooting.md` — Common issues and solutions
  - Model loading errors
  - Out-of-memory errors
  - Slow inference
  - Thermal throttling on Pi

## Reference Materials

- **`pi4b_specs.md`** — Raspberry Pi 4B hardware specifications & constraints (READ THIS FIRST!)
  - Processor (Quad-core ARM Cortex-A72)
  - Memory constraints (4GB total, ~3GB available)
  - Thermal limits, power consumption
  - Performance baselines
  - Optimization implications for VLA inference
  
- **`model_analysis.md`** — Deep analysis of Octo-small, π₀, and KV cache compression
  - Model architectures
  - Memory calculations
  - TurboQuant benefits
  - Comparison matrix

## Structure

```
docs/
├── README.md (this file)
├── understanding_plan.md       (learning roadmap)
├── turboquant_architecture.md  (Phase 1)
├── implementation_options.md   (Phase 1)
├── pi_benchmark_results.md     (Phase 2)
├── accuracy_validation.md      (Phase 2)
├── pi_profiling_report.md      (Phase 3)
├── turbopi_test_results.md     (Phase 4)
├── turbopi_deployment.md       (Phase 4)
├── performance_guide.md        (Phase 5)
├── troubleshooting.md          (Phase 5)
└── model_analysis.md           (reference)
```

## How to Use This Documentation

**Before starting**: Read `understanding_plan.md` to build mental model

**During Phase X**: Reference phase-specific deliverables

**For deployment**: Use `turbopi_deployment.md` and `troubleshooting.md`

**For tuning**: Use `performance_guide.md` and phase-specific benchmark results

**For reference**: Use `model_analysis.md` and `turboquant_architecture.md`

## Writing Documentation

When adding new docs:

1. **Use consistent format**:
   - Start with clear purpose statement
   - Include code examples where applicable
   - Add tables/diagrams for clarity
   - End with key takeaways or next steps

2. **Update this README**:
   - Add entry under appropriate phase or section
   - Include brief one-line description
   - Mark as "*(Pending)*" if not yet complete

3. **Location**:
   - Phase deliverables: `docs/` (this directory)
   - Design decisions: `docs/`
   - Code/module docs: Docstrings in `turboquant_kv/`, `octo_integration/`, etc.

## Document Status

| Document | Phase | Status | Priority |
|----------|-------|--------|----------|
| understanding_plan.md | Prerequisite | ✅ Done | Critical (read first) |
| turboquant_architecture.md | 1 | ⏳ TODO | High |
| implementation_options.md | 1 | ⏳ TODO | High |
| pi_benchmark_results.md | 2 | ⏳ TODO | High |
| accuracy_validation.md | 2 | ⏳ TODO | Medium |
| pi_profiling_report.md | 3 | ⏳ TODO | Medium |
| turbopi_test_results.md | 4 | ⏳ TODO | High |
| turbopi_deployment.md | 4 | ⏳ TODO | Critical (for users) |
| performance_guide.md | 5 | ⏳ TODO | Medium |
| troubleshooting.md | 5 | ⏳ TODO | High |

## Links to External Resources

- **TurboQuant Paper**: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **SGLang Implementation**: [github.com/sgl-project/sglang/pull/21617](https://github.com/sgl-project/sglang/pull/21617)
- **LeRobot Framework**: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- **Octo Models**: [HuggingFace](https://huggingface.co/google)
- **ONNX Runtime**: [onnxruntime.ai](https://onnxruntime.ai)

## Questions or Corrections?

If docs are unclear or incorrect:
1. Check related docs and code
2. Run examples to verify understanding
3. Note the issue for future updates

This documentation is a living document — it evolves as the project progresses.
