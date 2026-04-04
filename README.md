# TurboQuant VLA for Raspberry Pi 4B (TurboPi)

Deploy TurboQuant KV cache compression to enable real-time Vision-Language-Action (VLA) inference on **HiWonder TurboPi with Raspberry Pi 4B**.

**Hardware Target**: 
- **Processor**: Quad-core ARM Cortex-A72 (1.5 GHz)
- **Memory**: 4GB RAM
- **Goal**: 5-10 Hz inference latency with <5% accuracy loss

## Quick Links

- **[plan.md](plan.md)** — High-level overview and goals
- **[implementation_plan.md](implementation_plan.md)** — Detailed 5-phase roadmap (start here for implementation)
- **[CLAUDE.md](CLAUDE.md)** — Project guidance and collaboration guidelines
- **[docs/](docs/)** — Design documents, benchmarks, and deployment guides

## Project Status

| Phase | Task | Status |
|-------|------|--------|
| **Prerequisite** | Complete learning plan | 📖 [docs/understanding_plan.md](docs/understanding_plan.md) |
| **Phase 1** | Build TurboQuant core library | ⏳ TODO |
| **Phase 2** | Integrate with Octo-small, validate on Pi | ⏳ TODO |
| **Phase 3** | ONNX export & ARM optimization | ⏳ TODO |
| **Phase 4** | LeRobot integration & robot deployment | ⏳ TODO |
| **Phase 5** | Documentation & polish | ⏳ TODO |

## What's the Problem?

Vision-Language-Action models need 24GB+ VRAM due to KV cache growth. Raspberry Pi 4B has only 4GB RAM.

**Without optimization**: Model doesn't fit or runs at <1 Hz (unusable)

**With TurboQuant**: Fits in 4GB and runs at 5-10 Hz (real-time capable)

## The Solution

**TurboQuant**: Compress KV cache 3-4x using residual window quantization
- Recent tokens (last 512): Keep in FP16 (full precision, critical for context)
- Historical tokens: Quantize to INT4 (4x smaller, less important)
- Result: 37 MB KV cache → 12-15 MB (3x smaller)

## Target Hardware & Models

| Component | Spec |
|-----------|------|
| **Primary Target** | HiWonder TurboPi (Raspberry Pi 4B, 4GB RAM) |
| **Development** | RTX 4050 (8GB, simulate Pi constraints) |
| **Primary Model** | Octo-small (500M params) |
| **Backup Model** | π₀ quantized (3B params, pre-quantized) |

## Getting Started

### 1. Understand the Project (Prerequisite)

Read through in this order:
1. [plan.md](plan.md) — Overview and goals
2. [docs/understanding_plan.md](docs/understanding_plan.md) — **5-step learning path** (VLA basics → Octo-small → KV cache → TurboQuant → π₀)
3. [implementation_plan.md](implementation_plan.md) — Detailed roadmap
4. [CLAUDE.md](CLAUDE.md) — Collaboration guidelines

### 2. Set Up Development Environment

```bash
# Create uv environment
uv venv turboquant-vla-dev

# Activate
source turboquant-vla-dev/bin/activate  # macOS/Linux
# or
turboquant-vla-dev\Scripts\activate  # Windows

# Sync dependencies
uv sync

# Verify
python --version  # Should be 3.11+
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your settings
# - DEVICE: cpu, cuda, or mps
# - OCTO_MODEL_ID: Model to use
# - Other paths and configs
```

### 4. Start Phase 1 (Core Library)

Follow the implementation_plan.md:
- Task 1.1: Study TurboQuant paper (arXiv:2504.19874)
- Task 1.2: Analyze existing implementations
- Task 1.3: Build standalone TurboQuant library

## Directory Structure

```
turboquant-vla/
├── README.md                      # This file
├── CLAUDE.md                      # Project guidance for Claude
├── plan.md                        # High-level overview
├── implementation_plan.md         # Detailed roadmap (start here)
│
├── turboquant_kv/                # Core quantization library
│   ├── __init__.py
│   ├── quantizer.py              # KVQuantizer (TODO)
│   ├── calibration.py            # Calibrator (TODO)
│   ├── config.py                 # QuantizerConfig (TODO)
│   ├── utils.py                  # Helpers (TODO)
│   ├── kernels/                  # Optimized kernels
│   │   ├── __init__.py
│   │   └── quantized_attention.py # QuantizedAttention (TODO)
│   └── tests/                    # Unit tests
│       ├── __init__.py
│       ├── test_quantizer.py     # (TODO)
│       └── test_accuracy.py      # (TODO)
│
├── octo_integration/              # Octo model integration
│   ├── __init__.py
│   ├── octo_quantized.py         # OctoQuantizedInference (TODO)
│   └── tests/
│       └── test_octo_quantized.py # (TODO)
│
├── lerobot_integration/           # LeRobot integration
│   ├── __init__.py
│   ├── lerobot_policy.py         # LeRobotQuantizedPolicy (TODO)
│   └── tests/
│       └── test_lerobot_policy.py # (TODO)
│
├── scripts/                       # Benchmarks & demos
│   ├── README.md                 # Script documentation
│   ├── benchmark_octo_quantized.py     # (TODO: Phase 2)
│   ├── benchmark_octo_pi.py            # (TODO: Phase 2)
│   ├── benchmark_octo_pi_optimized.py  # (TODO: Phase 3)
│   ├── turbopi_demo.py                 # (TODO: Phase 4)
│   └── run_all_benchmarks.sh           # (TODO: Phase 5)
│
├── examples/                      # Working examples
│   ├── README.md                 # Example documentation
│   ├── octo_lerobot_inference.ipynb    # (TODO: Phase 2)
│   └── turbopi_demo.py                 # (TODO: Phase 4)
│
├── docs/                         # Documentation
│   ├── README.md                # Doc guide
│   ├── understanding_plan.md    # Learning path (read first!)
│   ├── turboquant_architecture.md      # (TODO: Phase 1)
│   ├── implementation_options.md       # (TODO: Phase 1)
│   ├── pi_benchmark_results.md         # (TODO: Phase 2)
│   ├── accuracy_validation.md          # (TODO: Phase 2)
│   ├── pi_profiling_report.md          # (TODO: Phase 3)
│   ├── turbopi_test_results.md         # (TODO: Phase 4)
│   ├── turbopi_deployment.md           # (TODO: Phase 4)
│   ├── performance_guide.md            # (TODO: Phase 5)
│   └── troubleshooting.md              # (TODO: Phase 5)
│
├── tests/                        # Integration tests
│   ├── __init__.py
│   ├── integration_test.py       # (TODO: Phase 4+)
│   └── accuracy_test.py          # (TODO: Phase 2+)
│
├── pyproject.toml               # Project metadata & uv config
├── .env.template                # Environment variables template
├── .gitignore                   # Git ignore rules
└── .github/                     # GitHub workflows (optional)
```

## Collaboration with Claude Code

This project uses **CLAUDE.md** for persistent project guidance. Key points:

- **Don't skip phases** — Each phase builds on the previous one
- **Test on real hardware** — Don't just simulate on RTX 4050
- **Benchmark early** — Profile often, optimize bottlenecks
- **Document design decisions** — Explain *why*, not just *what*

See [CLAUDE.md](CLAUDE.md) for full collaboration guidelines.

## Dependencies

**Core**:
- Python 3.9+
- PyTorch
- Transformers
- NumPy

**Optional**:
- ONNX Runtime (Phase 3+)
- LeRobot (Phase 4)
- Octo models (Phase 2)

**Development**:
- pytest (testing)
- black, ruff (code style)
- mypy (type checking)

Install all with:
```bash
uv sync
```

Install specific extras:
```bash
uv sync --extra dev     # Development tools
uv sync --extra octo    # Octo model support
uv sync --extra onnx    # ONNX Runtime
uv sync --extra pi      # ARM-optimized ONNX Runtime
```

## Common Commands

```bash
# Activate environment
source turboquant-vla-dev/bin/activate

# Run tests
pytest tests/ -v

# Run specific test
pytest turboquant_kv/tests/test_quantizer.py -v

# Run with coverage
pytest --cov=turboquant_kv

# Format code
black turboquant_kv/ octo_integration/ lerobot_integration/

# Lint code
ruff check turboquant_kv/

# Type checking
mypy turboquant_kv/

# Run benchmark
python scripts/benchmark_octo_quantized.py

# Run example
jupyter notebook examples/octo_lerobot_inference.ipynb
```

## Performance Targets

| Metric | Target | Phase |
|--------|--------|-------|
| **KV cache compression** | 3-4x | 1 |
| **Accuracy loss** | < 5% | 1 |
| **Inference latency** | 100-200 ms (5-10 Hz) | 2 |
| **Memory usage** | < 3.5 GB on Pi 4B | 2 |
| **ONNX optimization** | 10-20% speedup | 3 |
| **End-to-end** | Task execution on TurboPi | 4 |

## Resources

- **[TurboQuant Paper](https://arxiv.org/abs/2504.19874)** — arXiv:2504.19874
- **[SGLang Implementation](https://github.com/sgl-project/sglang/pull/21617)** — Reference implementation
- **[LeRobot Framework](https://github.com/huggingface/lerobot)** — VLA framework
- **[OpenVLA](https://github.com/openvla/openvla)** — Alternative VLA models
- **[ONNX Runtime](https://onnxruntime.ai)** — ARM-optimized inference

## Contributing & Feedback

- **Issue tracking**: See GitHub issues or CLAUDE.md
- **Collaboration**: See CLAUDE.md for guidelines
- **Questions**: Check docs/ or examples/ first

## License

MIT License — See LICENSE file (if present)

## Authors

**Suresh Gopalakrishnan** — Project lead

## Acknowledgments

- TurboQuant authors (arXiv:2504.19874)
- SGLang team (reference implementation)
- LeRobot community (VLA framework)
- HuggingFace (models and infrastructure)

---

**Status**: Planning phase → Learning prerequisite → Implementation  
**Last Updated**: 2026-04-04  
**Next Step**: Complete [docs/understanding_plan.md](docs/understanding_plan.md) learning path
